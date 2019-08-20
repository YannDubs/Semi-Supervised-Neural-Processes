import math

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, kl_divergence
import torch_scatter

from skssl.training.helpers import split_labelled_unlabelled
from skssl.utils.torchextend import (hellinger_dist, jensen_shannon_div,
                                     min_jensen_shannon_div, total_var)
from skssl.utils.helpers import input_to_graph

__all__ = ["NeuralProcessLoss", "GridNeuralProcessLoss", "GraphNeuralProcessLoss",
           "GridNeuralProcessSSLLoss", "GraphNeuralProcessSSLLoss", "NeuralProcessSSLLoss"]


class NeuralProcessLoss(nn.Module):
    """
    Compute the Neural Process Loss [1].

    Parameters
    ----------
    get_beta : callable, optional
        Function which returns the weight of the kl divergence at every call.

    is_sparse : bool, optional
        Whether the input is sparse multidimensional. If so the input shuld
        contain X_trgt.

    References
    ----------
    [1] Garnelo, Marta, et al. "Neural processes." arXiv preprint
        arXiv:1807.01622 (2018).
    """

    def __init__(self,
                 get_beta=lambda: 1,
                 is_sparse=False,
                 is_use_as_metric=False,
                 is_summary_loss=False,
                 ssl_loss=None,  # "supervised", "unsupervised", "both"
                 is_ssl_only=False,
                 distance="jsd",
                 n_max_elements=None,  # if given, used to to scale the loss
                 is_entropies=False,
                 is_consistency=True,
                 is_neg_consistency=True,
                 get_lambda_ent=lambda: 0.1,
                 get_lambda_unsup=lambda: 0.5,
                 get_lambda_sup=lambda: 1,
                 get_lambda_neg_cons=lambda: 0.2,
                 label_perc=None):   # if given, used to to scale the loss
        super().__init__()
        self.get_beta = get_beta
        self.is_use_as_metric = is_use_as_metric
        self.is_sparse = is_sparse
        self.is_summary_loss = is_summary_loss
        self.ssl_loss = ssl_loss
        self.get_lambda_sup = get_lambda_sup
        self.is_ssl_only = is_ssl_only
        self.distance = distance
        self.n_max_elements = n_max_elements
        self.is_entropies = is_entropies
        self.is_consistency = is_consistency
        self.label_perc = label_perc
        self.get_lambda_ent = get_lambda_ent
        self.get_lambda_unsup = get_lambda_unsup
        self.is_neg_consistency = is_neg_consistency
        self.get_lambda_neg_cons = get_lambda_neg_cons

        if self.ssl_loss in ["supervised", "both"]:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    def forward(self, inputs, y=None, weight=None):
        """Compute the Neural Process Loss averaged over the batch.

        Parameters
        ----------
        inputs: tuple
            Tuple of (p_y_trgt, Y_trgt, q_z_trgt, q_z_cntxt). This can directly
            take the output of NueralProcess.

        y: None
            Placeholder.

        weight: torch.Tensor, size = [batch_size,]
            Weight of every example. If None, every example is weighted by 1.
        """
        classifier_loss = 0

        if self.ssl_loss is not None:

            pred_logits = inputs[0]
            inputs = inputs[1:]
            pred_logits_lab, pred_logits_unlab = split_labelled_unlabelled(pred_logits, y)
            batch_size_unlab = pred_logits_unlab.size(0)
            batch_size = y.size(0)

            sup_loss = self.criterion(pred_logits, y)
            unsup_loss = 0

            if self.ssl_loss == "both" and batch_size_unlab != 0:

                pred_p_unlab = pred_logits_unlab.softmax(-1)
                p1 = pred_p_unlab[:batch_size_unlab // 2, ...]
                p2 = pred_p_unlab[batch_size_unlab // 2:, ...]

                if self.is_consistency:
                    if self.distance == "jsd":
                        dist = jensen_shannon_div(p1, p2)
                    elif self.distance == "totalvar":
                        dist = total_var(p1, p2)
                    else:
                        raise ValueError("Unkown dist =={}.".format(self.distance))

                    # if you do mean() you would not consider the fact that not all batch
                    # hase the same amount of labels
                    # if you divide by batch size, you would not consider the fact
                    # different expected count => divide by expected counts
                    unsup_loss = unsup_loss + dist.sum(0)

                    if self.is_neg_consistency:
                        # shift the probabilities by 1 and say that should be different
                        if self.distance == "jsd":
                            neg_dist = jensen_shannon_div(p1[1:], p2[:-1])
                        elif self.distance == "totalvar":
                            neg_dist = total_var(p1[1:], p2[:-1])
                        else:
                            raise ValueError("Unkown distance =={}.".format(self.distance))

                        unsup_loss = unsup_loss - self.get_lambda_neg_cons() * neg_dist.sum(0)

                if self.is_entropies:
                    entropies = Categorical(probs=p1).entropy() + Categorical(probs=p2).entropy()
                    unsup_loss = unsup_loss + entropies.sum(0) * self.get_lambda_ent()

            if self.label_perc is not None:
                unlabel_perc = 1 - self.label_perc
                classifier_loss = (self.get_lambda_sup() * sup_loss /
                                   (batch_size * self.label_perc) +
                                   self.get_lambda_unsup() * unsup_loss /
                                   (batch_size * unlabel_perc))
            else:
                classifier_loss = (self.get_lambda_sup() * sup_loss +
                                   self.get_lambda_unsup() * unsup_loss)

            if len(inputs) == 0:
                return sup_loss  # if validation only show supervised loss (beucase you are not batch duplicating cannot show unseupervised + no unsupervised in any case)

            if self.n_max_elements is not None:
                X_cntxt = inputs[6]
                n_cntxt = X_cntxt.size(1)
                classifier_loss = classifier_loss * n_cntxt / self.n_max_elements

            if self.is_ssl_only:
                return classifier_loss

        p_y_trgt, Y_trgt, q_z_trgt, q_z_cntxt, summary = inputs[:5]
        batch_size, n_trgt, _ = Y_trgt.shape

        if self.is_sparse:
            # not completely right because yo are averaging all, although
            # should compute batch average and then average over batches.
            X_trgt = inputs[5]
            channels = X_trgt[..., 0].int()

            n = 0
            for c in channels.unique():
                idcs = channels == c
                marginal_p_y_trgt = Normal(p_y_trgt.base_dist.loc[..., c:c + 1],
                                           p_y_trgt.base_dist.scale[..., c:c + 1])
                neg_log_like = - marginal_p_y_trgt.log_prob(Y_trgt)[idcs].sum()
                n += idcs.sum()
            neg_log_like = neg_log_like / n  # this already takes into account the batch average
            return neg_log_like
        else:
            # sum over the last dimension (y dim), then mean over poitns, and finally mean batch
            # neg_log_like = - p_y_trgt.log_prob(Y_trgt).sum(-1).view(batch_size, -1).mean(-1)
            neg_log_like = - p_y_trgt.log_prob(Y_trgt).view(batch_size, -1).mean(-1)

        if self.is_use_as_metric:
            # trick to use as metric => return log likelihood
            return - neg_log_like.mean(dim=0)

        if q_z_trgt is not None:
            # use latent variables and training
            # note that during validation the kl will actually be 0 because
            # we do not compute q_z_trgt
            kl_loss = kl_divergence(q_z_trgt, q_z_cntxt)
            # kl is multivariate Gaussian => sum over all target but we want mean
            kl_loss = kl_loss / n_trgt
        else:
            kl_loss = 0

        if self.is_summary_loss and summary is not None:  # don't add when scoring
            summary = summary.view(batch_size, -1)
            summaries_1 = summary[:batch_size // 2, ...]
            summaries_2 = summary[batch_size // 2:, ...]
            # positive samples : need to be the same => minimize diff
            pos_samples = (summaries_1 - summaries_2).abs().sum(-1).mean()
            # negative samples : mean needs to be different if not same sample => negative samples
            mean_summaries = (summaries_1 + summaries_2) / 2  # mean for all samples
            sum_summary = mean_summaries.sum(0, keepdim=True).expand(batch_size // 2, -1)  # sum of means
            mean_neg_samples = (sum_summary - mean_summaries) / (batch_size // 2 - 1)
            neg_samples = (mean_neg_samples - mean_summaries).abs().sum(-1).mean()
            # minimize positive diff and maximimize diff of negative samples
            # add a log(eps+x)*0.1 which says that as long as different from 1
            # you get some small gain but if less than 1 you get big losses
            summary_loss = pos_samples - torch.log(1e-5 + neg_samples) * 0.3
        else:
            summary_loss = 0

        loss = neg_log_like + self.get_beta() * kl_loss + summary_loss * 0.1

        if weight is not None:
            assert self.ssl_loss is None
            loss = loss * weight

        loss = loss.mean(dim=0) + classifier_loss

        if not torch.isfinite(loss):
            import pdb
            pdb.set_trace()

        return loss


class GridNeuralProcessLoss(nn.Module):
    """
    Compute the Neural Process Loss [1].

    Parameters
    ----------

    """

    def __init__(self,
                 is_use_as_metric=False,
                 is_ssl_only=False,
                 ssl_loss=None,  # "supervised",  "both"
                 n_max_elements=None,  # if given, used to to scale the loss
                 is_consistency=True,
                 is_neg_consistency=True,
                 is_entropies=False,
                 get_lambda_unsup=lambda: 0.5,
                 get_lambda_ent=lambda: 1.,
                 get_lambda_sup=lambda: 1,
                 get_lambda_neg_cons=lambda: 0.2,
                 label_perc=None,  # if given, used to to scale the loss
                 is_print=False):
        super().__init__()
        self.is_use_as_metric = is_use_as_metric
        self.ssl_loss = ssl_loss
        self.get_lambda_sup = get_lambda_sup
        self.n_max_elements = n_max_elements
        self.is_consistency = is_consistency
        self.label_perc = label_perc
        self.get_lambda_unsup = get_lambda_unsup
        self.get_lambda_ent = get_lambda_ent
        self.is_neg_consistency = is_neg_consistency
        self.get_lambda_neg_cons = get_lambda_neg_cons
        self.is_ssl_only = is_ssl_only
        self.is_entropies = is_entropies
        self.is_valid = True
        self.is_print = is_print

        if self.ssl_loss in ["supervised", "both"]:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    def forward(self, inputs, y=None, X=None, weight=None):
        """Compute the Neural Process Loss averaged over the batch.

        Parameters
        ----------
        inputs: tuple
            Tuple of (pred_logits?, p_y_trgt). This can directly
            take the output of NueralProcess.

        y: None
            Placeholder.

        weight: torch.Tensor, size = [batch_size,]
            Weight of every example. If None, every example is weighted by 1.
        """
        X, mask_context, mask_target = (X["X"],
                                        X["mask_context"].unsqueeze(-1),
                                        X["mask_target"].unsqueeze(-1))
        # puts channel in last dim
        X = X.permute(*([0] + list(range(2, X.dim())) + [1]))
        batch_size, *grid_shape, y_dim = X.shape

        classifier_loss = 0

        if self.ssl_loss is not None:

            pred_logits = inputs[0]
            inputs = inputs[1:]
            pred_logits_lab, pred_logits_unlab = split_labelled_unlabelled(pred_logits, y)
            batch_size_unlab = pred_logits_unlab.size(0)

            sup_loss = self.criterion(pred_logits, y)
            unsup_loss = 0
            neg_dist = 0
            dist = 0
            entropies = 0

            if self.label_perc is not None:
                unlabel_perc = 1 - self.label_perc
                denom_sup = self.label_perc * batch_size
                denom_unsup = unlabel_perc * batch_size
            else:
                denom_sup = denom_unsup = batch_size

            if self.ssl_loss == "both" and batch_size_unlab != 0:

                pred_p_unlab = pred_logits_unlab.softmax(-1)
                p1 = pred_p_unlab[:batch_size_unlab // 2, ...]
                p2 = pred_p_unlab[batch_size_unlab // 2:, ...]

                if self.is_consistency:
                    dist = jensen_shannon_div(p1, p2).sum(0)

                    # if you do mean() you would not consider the fact that not all batch
                    # hase the same amount of labels
                    # if you divide by batch size, you would not consider the fact
                    # different expected count => divide by expected counts
                    unsup_loss = unsup_loss + dist

                    if self.is_neg_consistency:
                        # shift the probabilities by 1 and say that should be different
                        neg_dist = jensen_shannon_div(p1[1:], p2[:-1]).sum(0)

                        unsup_loss = unsup_loss - self.get_lambda_neg_cons() * neg_dist

                if self.is_entropies:
                    entropies = Categorical(probs=pred_p_unlab).entropy().sum(0) / math.log(pred_logits.size(-1))
                    unsup_loss = unsup_loss + entropies * self.get_lambda_ent()

                unsup_loss = self.get_lambda_unsup() * unsup_loss / denom_unsup

            classifier_loss = (self.get_lambda_sup() * sup_loss / denom_sup +
                               unsup_loss)

            if len(inputs) == 0:
                self.is_valid = True
                return sup_loss / batch_size  # if validation only show supervised loss (beucase you are not batch duplicating cannot show unseupervised + no unsupervised in any case)

            if self.n_max_elements is not None:
                # asssue same across batch
                ratio_cntxt = (mask_context.view(batch_size, -1) != 0).sum(-1).float().mean() / self.n_max_elements
                classifier_loss = (classifier_loss * ratio_cntxt)

            if classifier_loss == 0:
                import pdb
                pdb.set_trace()

        if self.is_ssl_only:
            return classifier_loss

        p_y_trgt = inputs[0]

        Y_trgt = X.masked_select(mask_target).view(batch_size, -1, y_dim)

        neg_log_like = - p_y_trgt.log_prob(Y_trgt).view(batch_size, -1).mean(-1)

        if self.n_max_elements is not None:
            neg_log_like = (neg_log_like * (1 - ratio_cntxt))  # focus on this when low cntxt

        if self.is_use_as_metric:
            # trick to use as metric => return log likelihood
            return - neg_log_like.mean(dim=0)

        if weight is not None:
            neg_log_like = neg_log_like * weight

        loss = neg_log_like.mean(dim=0) + classifier_loss
        # loss = classifier_loss.mean(dim=0)

        if self.is_valid and self.is_print:
            with open('losses.logs', 'a') as f:
                print("neg_log_like", float(- p_y_trgt.log_prob(Y_trgt).view(batch_size, -1).mean(-1).mean(dim=0)),
                      "gnp_loss", float(neg_log_like.mean(dim=0)),
                      "classifier_loss", float(classifier_loss),
                      "sup_loss", float(sup_loss),
                      "neg_dist", float(neg_dist),
                      "entropies", float(entropies),
                      "dist", float(dist), file=f)
            self.is_valid = False

        if not torch.isfinite(loss):
            import pdb
            pdb.set_trace()

        return loss


class GraphNeuralProcessLoss(nn.Module):
    """
    Compute the Neural Process Loss [1].

    Parameters
    ----------

    """

    def __init__(self,
                 is_use_as_metric=False,
                 ssl_loss=None,  # "supervised",  "both"
                 n_max_elements=-1,  # if given, used to to scale the loss
                 is_consistency=True,
                 is_neg_consistency=True,
                 get_lambda_unsup=lambda: 0.5,
                 get_lambda_sup=lambda: 1,
                 get_lambda_neg_cons=lambda: 0.2,
                 label_perc=None,
                 is_ssl_only=False):   # if given, used to to scale the loss
        super().__init__()
        self.is_use_as_metric = is_use_as_metric
        self.ssl_loss = ssl_loss
        self.get_lambda_sup = get_lambda_sup
        self.n_max_elements = n_max_elements
        self.is_consistency = is_consistency
        self.label_perc = label_perc
        self.get_lambda_unsup = get_lambda_unsup
        self.is_neg_consistency = is_neg_consistency
        self.get_lambda_neg_cons = get_lambda_neg_cons
        self.is_valid = True
        self.is_ssl_only = is_ssl_only

        if self.ssl_loss in ["supervised", "both"]:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    def forward(self, inputs, y=None, X=None, weight=None):
        """Compute the Neural Process Loss averaged over the batch.

        Parameters
        ----------
        inputs: tuple
            Tuple of (pred_logits?, p_y_trgt). This can directly
            take the output of NueralProcess.

        y: None
            Placeholder.

        weight: torch.Tensor, size = [batch_size,]
            Weight of every example. If None, every example is weighted by 1.
        """

        data, mask_context, mask_target = (input_to_graph(X["X"]),
                                           X["mask_context"].unsqueeze(-1),
                                           X["mask_target"].unsqueeze(-1))
        classifier_loss = 0

        if self.ssl_loss is not None:

            batch_size = len(y)
            pred_logits = inputs[0]
            inputs = inputs[1:]
            pred_logits_lab, pred_logits_unlab = split_labelled_unlabelled(pred_logits, y)
            batch_size_unlab = pred_logits_unlab.size(0)

            sup_loss = self.criterion(pred_logits, y)
            unsup_loss = 0
            neg_dist = 0
            dist = 0

            if self.label_perc is not None:
                unlabel_perc = 1 - self.label_perc
                denom_sup = self.label_perc * batch_size
                denom_unsup = unlabel_perc * batch_size
            else:
                denom_sup = denom_unsup = batch_size

            if self.ssl_loss == "both" and batch_size_unlab != 0:

                pred_p_unlab = pred_logits_unlab.softmax(-1)
                p1 = pred_p_unlab[:batch_size_unlab // 2, ...]
                p2 = pred_p_unlab[batch_size_unlab // 2:, ...]

                if self.is_consistency:
                    dist = jensen_shannon_div(p1, p2).sum(0)

                    # if you do mean() you would not consider the fact that not all batch
                    # hase the same amount of labels
                    # if you divide by batch size, you would not consider the fact
                    # different expected count => divide by expected counts
                    unsup_loss = unsup_loss + dist

                    if self.is_neg_consistency:
                        # shift the probabilities by 1 and say that should be different
                        neg_dist = jensen_shannon_div(p1[1:], p2[:-1]).sum(0)

                        unsup_loss = unsup_loss - self.get_lambda_neg_cons() * neg_dist

                unsup_loss = self.get_lambda_unsup() * unsup_loss / denom_unsup

            classifier_loss = (self.get_lambda_sup() * sup_loss / denom_sup +
                               unsup_loss)

            if len(inputs) == 0:
                self.is_valid = True
                return sup_loss / batch_size  # if validation only show supervised loss (beucase you are not batch duplicating cannot show unseupervised + no unsupervised in any case)

            if self.n_max_elements == -1:
                ratio_cntxt = mask_context.float().mean()  # asssue same across batch
                classifier_loss = classifier_loss * ratio_cntxt
            else:
                assert self.n_max_elements is None

            if classifier_loss == 0:
                import pdb
                pdb.set_trace()

        if self.is_ssl_only:
            return classifier_loss

        p_y_trgt = inputs[0]

        Y_trgt = data.x[mask_target.squeeze(-1), :]
        batch_trgt = data.batch[mask_target.squeeze(-1)]

        neg_log_like = - p_y_trgt.log_prob(Y_trgt).sum(-1)  # sum over features
        neg_log_like = torch_scatter.scatter_mean(neg_log_like, batch_trgt)  # mean over points

        if self.n_max_elements is not None:
            neg_log_like = (neg_log_like * (1 - ratio_cntxt))  # focus on this when low cntxt

        if self.is_use_as_metric:
            # trick to use as metric => return log likelihood
            return - neg_log_like.mean(dim=0)

        if weight is not None:
            neg_log_like = neg_log_like * weight

        loss = neg_log_like.mean(dim=0) + classifier_loss

        if self.is_valid:
            print("neg_log_like", float(torch_scatter.scatter_mean(- p_y_trgt.log_prob(Y_trgt).sum(-1), batch_trgt).mean(dim=0)),
                  "gnp_loss", float(neg_log_like.mean(dim=0)),
                  "classifier_loss", float(classifier_loss),
                  "sup_loss", float(sup_loss),
                  "neg_dist", float(neg_dist),
                  "dist", float(dist))
            self.is_valid = False

        if not torch.isfinite(loss):
            import pdb
            pdb.set_trace()

        return loss


class GridNeuralProcessSSLLoss(nn.Module):

    def __init__(self,
                 n_max_elements=None,
                 get_lambda_unsup=lambda: 1,
                 get_lambda_ent=lambda: 0.5,  # both do something similar
                 get_lambda_sup=lambda: 5,
                 get_lambda_neg_cons=lambda: 0.5,  # both do something similar
                 label_perc=None,
                 min_sigma=0.1,
                 is_clf=True,
                 is_ssl_only=False,
                 is_unsup_forall=False,
                 is_print=False):
        super().__init__()

        self.n_max_elements = n_max_elements
        self.label_perc = label_perc
        self.is_clf = is_clf
        self.min_sigma = min_sigma

        self.get_lambda_sup = get_lambda_sup
        self.get_lambda_unsup = get_lambda_unsup
        self.get_lambda_ent = get_lambda_ent
        self.get_lambda_neg_cons = get_lambda_neg_cons
        self.is_ssl_only = is_ssl_only
        self.is_unsup_forall = is_unsup_forall

        self.is_print = is_print
        self.is_valid = True

        if self.is_clf:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    def forward(self, inputs, y=None, X=None):
        """Compute the Neural Process Loss averaged over the batch.

        Parameters
        ----------
        inputs: tuple
            Tuple of (pred_logits?, p_y_trgt). This can directly
            take the output of NueralProcess.

        y: None
            Placeholder.

        weight: torch.Tensor, size = [batch_size,]
            Weight of every example. If None, every example is weighted by 1.
        """
        preprocessed_X = self.preprocess_X(X)

        if self.is_clf:
            pred_logits = inputs[0]
            inputs = inputs[1:]
            sup_loss = self.compute_sup_loss(pred_logits, y)

            if len(inputs) == 0:
                # VALIDATION
                self.is_valid = True
                return sup_loss / len(y)

            unsup_loss, unsup_print = self.compute_unsup_loss(pred_logits, y)
            clf_loss = self.scale_clf_losses(sup_loss, unsup_loss, pred_logits, **preprocessed_X)

        else:
            clf_loss = unsup_loss = sup_loss = unsup_print = 0

        if self.is_ssl_only:
            gnp_loss = nll = 0
        else:
            p_y_trgt = inputs[0]
            nll = self.get_nll(p_y_trgt, **preprocessed_X)
            gnp_loss = self.scale_gnp_losses(nll, p_y_trgt, **preprocessed_X)

        loss = gnp_loss + clf_loss

        if self.is_print and self.is_valid:
            print("nll", float(nll), "gnp_loss", float(gnp_loss), "clf_loss", float(clf_loss),
                  "sup_loss", float(sup_loss), "unsup_loss", float(unsup_loss),
                  "unsup_print", unsup_print)
            self.is_valid = False

        if not torch.isfinite(loss):
            import pdb
            pdb.set_trace()

        return loss

    def scale_gnp_losses(self, nll, p_y_trgt, **preprocessed_X):
        y_dim = preprocessed_X["X"].size(-1)
        # negative normalized log likelihood
        nll = nll / (-math.log((self.min_sigma * math.sqrt(2 * math.pi))
                               ) * y_dim)

        if self.n_max_elements is not None:
            ratio_cntxt = self.get_ratio_cntxt(**preprocessed_X)
            nll = (nll * (1 - ratio_cntxt))

        return nll

    def get_nll(self, p_y_trgt, X, mask_context, mask_target):

        batch_size, *grid_shape, y_dim = X.shape

        Y_trgt = X.masked_select(mask_target).view(batch_size, -1, y_dim)

        nll = - p_y_trgt.log_prob(Y_trgt).view(batch_size, -1).mean(-1)

        return nll.mean(0)

    def preprocess_X(self, X):
        X, mask_context, mask_target = (X["X"],
                                        X["mask_context"].unsqueeze(-1),
                                        X["mask_target"].unsqueeze(-1))
        # puts channel in last dim
        X = X.permute(*([0] + list(range(2, X.dim())) + [1]))
        return dict(X=X, mask_context=mask_context, mask_target=mask_target)

    def compute_sup_loss(self, pred_logits, y):
        return self.criterion(pred_logits, y)

    def compute_unsup_loss(self, pred_logits, y):

        if self.is_unsup_forall:
            pred_logits_unlab = pred_logits
        else:
            pred_logits_lab, pred_logits_unlab = split_labelled_unlabelled(pred_logits, y)

        batch_size_unlab = pred_logits_unlab.size(0)

        if batch_size_unlab != 0:

            pred_p_unlab = pred_logits_unlab.softmax(-1)
            p1 = pred_p_unlab[:batch_size_unlab // 2, ...]
            p2 = pred_p_unlab[batch_size_unlab // 2:, ...]

            # make max be 1 (divide by ln(2) and times 2 because you are suming
            # onyl half of the batche but dividing by all batch size
            dist = jensen_shannon_div(p1, p2).sum(0) * 2 / math.log(2)

            # invert the second btch and say that should be different
            neg_dist = - jensen_shannon_div(p1, torch.flip(p2, [0])).sum(0) * 2 / math.log(2)

            # divide by ln(y_dim) to be max 1

            entropies = (Categorical(probs=pred_p_unlab).entropy().sum(0) /
                         math.log(pred_logits.size(-1)))

            unsup_loss = (dist +
                          entropies * self.get_lambda_ent() +
                          neg_dist * self.get_lambda_neg_cons())

            print_unsup = dict(dist=float(dist),
                               entropies=float(entropies),
                               neg_dist=float(neg_dist),
                               max_pred=pred_p_unlab.max())

        else:
            unsup_loss = 0
            print_unsup = 0

        return unsup_loss, print_unsup

    def scale_clf_losses(self, sup_loss, unsup_loss, pred_logits, **preprocessed_X):

        batch_size = pred_logits.size(0)

        if self.label_perc is None or self.label_perc == 1:
            denom_unsup = denom_sup = batch_size
        else:
            unlabel_perc = 1 - self.label_perc if not self.is_unsup_forall else 1
            denom_sup = self.label_perc * batch_size
            denom_unsup = unlabel_perc * batch_size

        # if you do mean() you would not consider the fact that not all batch
        # hase the same amount of labels
        # if you divide by batch size, you would not consider the fact
        # different expected count => divide by expected counts

        clf_loss = (self.get_lambda_sup() * sup_loss / denom_sup +
                    self.get_lambda_unsup() * unsup_loss / denom_unsup)

        if self.n_max_elements is not None:
            ratio_cntxt = self.get_ratio_cntxt(**preprocessed_X)
            clf_loss = (clf_loss * ratio_cntxt)

        return clf_loss

    def get_ratio_cntxt(self, X, mask_context, mask_target):
        batch_size = X.size(0)

        # asssue same across batch
        ratio_cntxt = ((mask_context.view(batch_size, -1) != 0).sum(-1).float().mean() /
                       self.n_max_elements)

        return ratio_cntxt


class GraphNeuralProcessSSLLoss(GridNeuralProcessSSLLoss):

    def get_nll(self, p_y_trgt, X, mask_context, mask_target):

        Y_trgt = X.x[mask_target.squeeze(-1), :]
        batch_trgt = X.batch[mask_target.squeeze(-1)]

        nll = - p_y_trgt.log_prob(Y_trgt).sum(-1)  # sum over features
        nll = torch_scatter.scatter_mean(nll, batch_trgt)  # mean over points in batches
        nll = nll.mean()  # mean over batch

        return nll

    def preprocess_X(self, X):
        X, mask_context, mask_target = (input_to_graph(X["X"]),
                                        X["mask_context"].unsqueeze(-1),
                                        X["mask_target"].unsqueeze(-1))
        return dict(X=X, mask_context=mask_context, mask_target=mask_target)

    def get_ratio_cntxt(self, X, mask_context, mask_target):
        return mask_context.float().mean()

    def scale_gnp_losses(self, nll, p_y_trgt, **preprocessed_X):
        # negative normalized log likelihood
        y_dim = preprocessed_X["X"].x.size(-1)
        nll = nll / (-math.log((self.min_sigma * math.sqrt(2 * math.pi))
                               ) * y_dim)

        if self.n_max_elements is not None:
            ratio_cntxt = self.get_ratio_cntxt(**preprocessed_X)
            nll = (nll * (1 - ratio_cntxt))

        return nll


class NeuralProcessSSLLoss(GridNeuralProcessSSLLoss):

    def get_nll(self, p_y_trgt, X_cntxt, Y_cntxt, X_trgt, Y_trgt):

        batch_size, n_trgt, _ = Y_trgt.shape
        nll = - p_y_trgt.log_prob(Y_trgt  # sum over features
                                  ).view(batch_size, -1).mean(-1)  # mean over points in batches
        nll = nll.mean()  # mean over batch

        return nll

    def preprocess_X(self, X):
        X_cntxt, Y_cntxt, X_trgt, Y_trgt = X["X"], X["y"], X["X_trgt"], X["y_trgt"]
        return dict(X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, Y_trgt=Y_trgt)

    def get_ratio_cntxt(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt):
        n_cntxt = X_cntxt.size(1)
        ratio_cntxt = n_cntxt / self.n_max_elements
        return ratio_cntxt

    def scale_gnp_losses(self, nll, p_y_trgt, **preprocessed_X):
        # negative normalized log likelihood
        y_dim = preprocessed_X["Y_trgt"].size(-1)
        nll = nll / (-math.log((self.min_sigma * math.sqrt(2 * math.pi))
                               ) * y_dim)

        if self.n_max_elements is not None:
            ratio_cntxt = self.get_ratio_cntxt(**preprocessed_X)
            nll = (nll * (1 - ratio_cntxt))

        return nll
