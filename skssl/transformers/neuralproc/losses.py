import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, kl_divergence

from skssl.training.helpers import split_labelled_unlabelled

__all__ = ["NeuralProcessLoss"]


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

    def __init__(self, get_beta=lambda: 1,
                 is_sparse=False,
                 is_use_as_metric=False,
                 is_summary_loss=False,
                 ssl_loss=None):  # "supervised", "unsupervised", "both"
        super().__init__()
        self.get_beta = get_beta
        self.is_use_as_metric = is_use_as_metric
        self.is_sparse = is_sparse
        self.is_summary_loss = is_summary_loss
        self.ssl_loss = ssl_loss

        if self.ssl_loss in ["supervised", "both"]:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

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
        if self.ssl_loss is not None:
            pred_logits = inputs[0]
            inputs = inputs[1:]
            pred_logits_lab, pred_logits_unlab = split_labelled_unlabelled(pred_logits, y)

            if self.ssl_loss in ["supervised", "both"]:
                sup_loss = self.criterion(pred_logits, y)

            if self.ssl_loss in ["unsupervised", "both"]:
                batch_size_unlab = pred_logits_unlab.size(0)
                p1 = pred_logits_unlab[:batch_size_unlab // 2, ...]
                p2 = pred_logits_unlab[batch_size_unlab // 2:, ...]
                unsup_loss = (kl_divergence(p1, p2) + kl_divergence(p2, p1)) / 2  # make symmetric
        else:
            sup_loss = 0
            unsup_loss = 0

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
            #neg_log_like = - p_y_trgt.log_prob(Y_trgt).sum(-1).view(batch_size, -1).mean(-1)
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

        loss = (neg_log_like + self.get_beta() * kl_loss + summary_loss * 0.1 +
                sup_loss + unsup_loss)

        if weight is not None:
            loss = loss * weight

        return loss.mean(dim=0)
