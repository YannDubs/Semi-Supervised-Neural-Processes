"""
Framework for semi supervised learning VAE from:

Kingma, Durk P., et al. "Semi-supervised learning with deep generative
models." Advances in neural information processing systems. 2014.
"""

import torch

import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import ClassifierMixin, TransformerMixin

from econvcnp.transformers import VAELoss
from econvcnp.predefined import MLP, WideResNet, ReversedWideResNet, merge_flat_input
from econvcnp.utils.initialization import weights_init
from econvcnp.utils.torchextend import reparameterize
from econvcnp.utils.helpers import cont_tuple_to_tuple_cont
from econvcnp.training.helpers import split_labelled_unlabelled


__all__ = ['SSLVAELoss', 'SSLVAE']


class SSLVAELoss(VAELoss):
    """
    Compute the SSL VAE loss as in [1].

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence. If 1, corresponds to standard VAE.

    kwargs:
        Additional arguments to `VAELoss.

    References
    ----------
        [1] Kingma, Durk P., et al. "Semi-supervised learning with deep generative
        models." Advances in neural information processing systems. 2014.
    """

    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.criterion_y = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    def _split_inputs(self, inputs):
        """Split the inputs."""
        pred_logits, vae_inputs_lab, vae_inputs_unlab = inputs[0], inputs[1:4], inputs[4:7]
        add_out = inputs[7] if len(inputs) > 7 else {}
        return pred_logits, vae_inputs_lab, vae_inputs_unlab, add_out

    def forward(self, inputs, y=None, X=None):
        """Compute the SSL VAE loss.

        Note
        ----
        - Doesn't compute the loss due to p(y) because doesn't depend on param

        Parameters
        ----------
        inputs : tuple
            Tuple of (pred_logits, *vae_inputs_lab, *vae_inputs_unlab, add_out).
            This can directly take the output of SSLVAE.

        y : torch.Tensor, size = [batch_size]
            Labels. -1 for unlabelled. `None` if all unlabelled.

        X : torch.Tensor, size = [batch_size, *x_shape]
            Training data corresponding to the targets.
        """
        assert X is not None
        assert y is not None

        pred_logits, vae_inputs_lab, vae_inputs_unlab, add_out = self._split_inputs(inputs)
        add_out_lab, add_out_unlab = split_labelled_unlabelled(add_out, y)
        X_lab, X_unlab = split_labelled_unlabelled(X, y)
        pred_logits_lab, pred_logits_unlab = split_labelled_unlabelled(pred_logits, y)
        n_lab, n_unlab = pred_logits_lab.size(0), pred_logits_unlab.size(0)
        labelled_loss = unlabelled_loss = 0

        if n_lab != 0:
            labelled_loss = self._labelled_loss(vae_inputs_lab, X_lab, **add_out_lab)

        if n_unlab != 0:
            unlabelled_loss = self._unlabelled_loss(vae_inputs_unlab, X_unlab,
                                                    pred_logits_unlab, **add_out_unlab)

        # - log q(y|x)
        classifier_loss = self.criterion_y(pred_logits, y)

        return labelled_loss + unlabelled_loss + self.alpha * classifier_loss

    def _unlabelled_loss(self, vae_inputs, X, pred_logits, **kwargs):
        """for unlabelled data: E_y [labelled loss] - H[q(y|x)]"""
        p_y_hat = F.softmax(pred_logits, dim=-1)
        n_unlab, y_dim = pred_logits.shape

        def _get_vae_inputs(i):
            return tuple(inp[:, i, ...] for inp in vae_inputs)

        expected_lab_loss = sum(self._labelled_loss(_get_vae_inputs(i), X,
                                                    weight=p_y_hat[:, i],
                                                    **kwargs)
                                for i in range(y_dim))

        # H[q(y|x)] = -dot(q,log(q))
        ent_y = - torch.bmm(p_y_hat.view(-1, 1, y_dim),
                            F.log_softmax(pred_logits, dim=-1).view(-1, y_dim, 1)
                            ).mean()

        return expected_lab_loss - ent_y

    def _labelled_loss(self, vae_inputs, X, weight=None, **kwargs):
        """for labelled data: - log p(x|y,z) + beta KL(q(z|x,y)||p(z))"""
        return super().forward(vae_inputs, X=X, weight=weight)


# MODEL
class SSLVAE(nn.Module):
    """Semisupervised VAE from [1] (M2 model). This is a classifier, so the predicted
    label is the output of `NeuralNet.predict`. Using `NeuralNet.transform` will
    return the latent representation. `SSLVAE.sample_decode` can still be used
    for generation.

    Parameters
    ----------
    x_shape: array-like
        Shape of a single example x.

    y_dim: int
        Number of classes.

    z_dim: int, optional
        Number of latent dimensions.

    transform_dim: int, optional
        Number of dimension after applying a differential transform on the input
        x. If `None` does not apply any transforms. This is very useful for high
        dimesnional non flat inputs (like images) such that the encoder and decoder
        only use a low dimensional input. The tranformed vector will be passed through
        a relu activation before encoder / classifier (so not 2 linear layers in a row)

    Encoder: nn.Module, optional
        Encoder module which maps X(_transf), y -> z_suff_stat. It should be callable
        with `encoder(x_shape, y_dim, n_out)`. If you have an encoder that maps
        x -> z you can convert it via `merge_flat_input(Encoder)`.

    Decoder: nn.Module, optional
        Decoder module which maps [z;y] -> X. It should be callable with
        `decoder(z_dim, x_shape)`. No non linearities should be applied to the
        output.

    Classifier: nn.Module, optional
        Classifier module which mapy X(_transf) -> y. The last layer should be a softmax.

    Transformer: nn.Module, optional
        Transformer module which maps X -> X_transf. Only used if `transform_dim` is not
        None. See `transform_dim` for more details.

    Reference
    ---------
    [1] Kingma, D. P., Mohamed, S., Rezende, D. J., & Welling, M. (2014).
        Semi-supervised learning with deep generative models. In Advances in neural
        information processing systems (pp. 3581-3589).
    """

    def __init__(self, x_shape, y_dim,
                 z_dim=64,
                 transform_dim=None,
                 Encoder=merge_flat_input(MLP),
                 Decoder=ReversedWideResNet,
                 Classifier=MLP,
                 Transformer=WideResNet):

        super().__init__()
        self.x_shape = x_shape
        self.transform_dim = transform_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.transform_dim = transform_dim

        if self.transform_dim is not None:
            self.transformer = Transformer(self.x_shape, self.transform_dim)
        else:
            transform_dim = x_shape

        self.classifier = Classifier(transform_dim, self.y_dim)
        self.encoder = Encoder(transform_dim, self.y_dim, self.z_dim * 2)
        self.decoder = Decoder(self.y_dim + self.z_dim, self.x_shape)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X, y=None):
        """
        Forward pass of model.

        Parameters
        ----------
        X : torch.Tensor, size = [batch_size, *x_shape]
            Batch of data.

        y : torch.Tensor, size = [batch_size]
            Labels. -1 for unlabelled. `None` if all unlabelled.

        Returns
        -------
        pred_logits: torch.Tensor, size = [batch_size, y_dim]
            Multinomial logits (i.e. no softmax).

        z_sample_lab: torch.Tensor, size = [n_lab, z_dim]
            Latent sample of labelled data.

        z_suff_stat_lab: torch.Tensor, size = [n_lab, z_dim*2]
            Sufficient statistics of the labelled latent sample {mu; logvar}.

        reconstruct_lab: torch.Tensor, size = [n_lab, *x_shape]
            Reconstructed labelled image. Values between 0,1 (i.e. after logistic).

        z_sample_unlab: torch.Tensor, size = [n_unlab, y_dim, z_dim]
            Latent sample of labelled data. Stacked on first dimension for
            each possible y (in order).

        z_suff_stat_unlab: torch.Tensor, size = [n_unlab, y_dim, z_dim*2]
            Sufficient statistics of the labelled latent sample {mu; logvar}.
            Stacked on first dimension for each possible y (in order).

        reconstruct_unlab: torch.Tensor, size = [n_unlab, y_dim, *x_shape]
            Reconstructed labelled image. Values between 0,1 (i.e. after logistic).
            Stacked on first dimension for each possible y (in order).
        """
        device = X.device
        batch_size = X.size(0)

        if y is None:
            y = torch.tensor([-1], device=device).expand(batch_size)

        if self.transform_dim is not None:
            X = self.transformer(X)
            X = torch.relu(X)  # make sure not 2 linear layer in a row

        # following line is not used in sslvae but might be in classes that
        # inherit from it (E.g. auxsslvae)
        add_out = self._get_additional_outputs(X)

        X_lab, X_unlab = split_labelled_unlabelled(X, y)
        y_lab, y_unlab = split_labelled_unlabelled(y, y)
        add_out_lab, add_out_unlab = split_labelled_unlabelled(add_out, y)
        n_lab, n_unlab = y_lab.size(0), y_unlab.size(0)

        pred_logits = self.classify(X, **add_out)

        assert n_lab != 0 or n_unlab != 0
        if n_lab != 0:
            y_lab_onehot = torch.zeros_like(pred_logits[:n_lab, ...]
                                            ).scatter_(1, y_lab.view(-1, 1), 1)
            out_lab = self._forward_labelled(X_lab, y_lab_onehot, **add_out_lab)

        if n_unlab != 0:
            # no copying
            y_unlab_onehot = torch.zeros_like(pred_logits[:1, ...]).expand(n_unlab, self.y_dim)
            out_unlab = self._forward_unlabelled(X_unlab, y_unlab_onehot, **add_out_unlab)

        # make sure same numebr of outputs (even if empty)
        if n_lab == 0:
            out_lab = tuple(torch.tensor([]) for _ in out_unlab)
        if n_unlab == 0:
            out_unlab = tuple(torch.tensor([]) for _ in out_lab)

        out = (pred_logits,) + out_lab + out_unlab
        if len(add_out) > 0:
            out += (add_out,)

        # see outputs in docstrings
        return out

    def _get_additional_outputs(self, X):
        """Function that can be overriden to add outputs to the forward method."""
        return {}

    def classify(self, X, **kwargs):
        """Classify the given X by returning multinomial logits."""
        pred_logits = self.classifier(X)
        return pred_logits

    def reparameterize(self, z_suff_stat):
        """Sample from gaussian with reparameterization trick."""
        return reparameterize(z_suff_stat, is_sample=self.training)

    def _forward_labelled(self, X, y_onehot, **kwargs):
        """Forward fucntion for labbeled examples."""
        z_suff_stat = self.encoder(X, y_onehot)
        z_sample = self.reparameterize(z_suff_stat)
        reconstruct = torch.sigmoid(self.decoder(torch.cat((z_sample, y_onehot), dim=1)))
        return z_sample, z_suff_stat, reconstruct

    def _forward_unlabelled(self, X, y_onehot, **kwargs):
        """Same as for labelled but want to marginalize out labels => compute outputs for all y"""
        def _get_y_onehot(i):
            y_onehot.zero_()
            y_onehot[:, i] = 1
            return y_onehot
        # [(out1_l1,out2_l1,...), (out1_l2,out2_l2,...)]
        out = [self._forward_labelled(X, _get_y_onehot(l), **kwargs)
               for l in range(self.y_dim)]
        # ([out1_l1, out1_l2, ...], [out2_l1, out2_l2, ...])
        out = cont_tuple_to_tuple_cont(out)
        # stack the list of tensors on first dim
        out = tuple(torch.stack(o, dim=1) for o in out)
        return out

    def sample_decode(self, z, y):
        """
        Returns a sample from the decoder.

        Parameters
        ----------
        z : torch.Tensor, size = [batch_size, z_dim]
            Latent variable.

        y : torch.Tensor, size = [batch_size]
            Labels.
        """
        batch_size = z.size(0)
        y_onehot = torch.zeros(batch_size, self.y_dim, device=z.device).float()
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        sample = torch.sigmoid(self.decoder(torch.cat((z, y_onehot), dim=1)))
        return sample
