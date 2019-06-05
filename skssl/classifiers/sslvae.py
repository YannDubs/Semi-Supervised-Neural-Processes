"""
Framework for semi supervised learning VAE from:

Kingma, Durk P., et al. "Semi-supervised learning with deep generative
models." Advances in neural information processing systems. 2014.
"""

import torch

import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import ClassifierMixin, TransformerMixin

from skssl.transformers import VAELoss
from skssl.predefined import MLP, WideResNet, ReversedSimpleCNN
from skssl.predefined.helpers import add_flat_input
from skssl.utils.initialization import weights_init
from skssl.training.helpers import split_labelled_unlabelled

__all__ = ['SSLVAELoss', 'SSLVAE']


class SSLVAELoss(VAELoss):
    """
    Compute the SSL VAE loss as in [1].

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence. If 1, corresponds to standard VAE.

    distribution : {"bernoulli", "gaussian", "laplace"}, optional
        Distribution of the likelihood for each feature. See `reconstruction_loss`
        for more details.

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

    def forward(self, inputs, y=None, X=None):
        """Compute the SSL VAE loss.

        Note
        ----
        - Doesn't compute the loss due to p(y) because doesn't depend on param

        Parameters
        ----------
        inputs : tuple
            Tuple of (y_hat, z_sample, reconstruct, *). This can directly take the output
            of VAE.

        y : None
            Placeholder.

        X : torch.Tensor, size = [batch_size, *x_shape]
            Training data corresponding to the targets.
        """
        assert X is not None
        assert y is not None

        y_hat = inputs[0]

        # reverts back to have (y_hat, reconstruct, z_sample, *) like for VAELoss
        inputs = [inputs[2] + inputs[1]] + inputs[3:]

        X_lab, X_unlab = split_labelled_unlabelled(X, y)
        # stacked in dim=1 : the first n_lab are the labelled ones and the rest are unlabelled
        inputs_lab, inputs_unlab = split_labelled_unlabelled(inputs, None, is_stacked=True)
        labelled_loss = unlabelled_loss = 0

        if (y != -1).sum() != 0:
            labelled_loss = self._labelled_loss(inputs_lab, X_lab)
        if (y == -1).sum() != 0:
            unlabelled_loss = self._unlabelled_loss(inputs_unlab, X_unlab, y_hat)

        # - log q(y|x)
        classifier_loss = self.criterion_y(y_hat, y)

        return labelled_loss + unlabelled_loss + self.alpha * classifier_loss

    def _unlabelled_loss(self, vae_inputs, X, y_hat):
        """for unlabelled data: E_y [labelled loss] - H[q(y|x)]"""
        p_y_hat = F.softmax(y_hat, dim=-1)
        n_unlab, y_dim = y_hat.shape

        expected_lab_loss = sum(self._labelled_loss(vae_inputs[:, i, ...], X, weight=p_y_hat[:, i])
                                for i in range(y_dim))
        # H[q(y|x)] = -dot(q,log(q))
        ent_y = - torch.bmm(p_y_hat.view(-1, 1, y_dim),
                            F.log_softmax(y_hat, dim=-1).view(-1, y_dim, 1)
                            ).mean()
        return expected_lab_loss - ent_y

    def _labelled_loss(self, vae_inputs, X, **kwargs):
        """for labelled data: - log p(x|y,z) + beta KL(q(z|x,y)||p(z))"""
        return super().forward(vae_inputs, X=X, **kwargs)


# MODEL
class SSLVAE(nn.Module, TransformerMixin, ClassifierMixin):
    """Semisupervised VAE from [1] (M2 model). This is a classifier, so the predicted
    label is the output of `NeuralNet.predict`. Using `NeuralNet.transform` will
    return the latent representation. `SSLVAE.sample_decode` can still be used
    for generation.

    Parameters
    ----------
    x_shape : array-like
        Shape of a single example x.

    y_dim : int
        Number of classes.

    z_dim : int, optional
        Number of latent dimensions.

    Encoder : nn.Module, optional
        Encoder module which maps x,y -> z. It should be callable with
        `encoder(x_shape, y_dim, n_out)`. If you have an encoder that maps x -> z
        you convert it via `add_flat_input(Encoder)`.

    Decoder : nn.Module, optional
        Decoder module which maps z,y -> x. It should be callable with
        `decoder(z_dim, x_shape)`. No non linearities should be applied to the
        output.

    Classifier : nn.Module, optional
        Classifier module which mapy X -> y. The last layer should be a softmax.

    Reference
    ---------
    [1] Kingma, D. P., Mohamed, S., Rezende, D. J., & Welling, M. (2014).
        Semi-supervised learning with deep generative models. In Advances in neural
        information processing systems (pp. 3581-3589).
    """

    def __init__(self, x_shape, y_dim, z_dim=10,
                 Encoder=add_flat_input(WideResNet),
                 Decoder=ReversedSimpleCNN,
                 Classifier=MLP):

        super().__init__()
        self.x_shape = x_shape
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.encoder = Encoder(x_shape, y_dim, z_dim * 2)
        self.decoder = Decoder(y_dim + z_dim, x_shape)
        self.classifier = Classifier(x_shape, y_dim)

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
        ------
        y_hat: torch.Tensor, size = [batch_size, y_dim]
            Multinomial logits (i.e. no softmax).

        z_sample: torch.Tensor, size = [batch_size, y_dim+1, z_dim]
            Latent sample. The first dimension corresponds to [labbeled] +
            [all possible values of y when unlabelled]

        reconstruct: torch.Tensor, size = [batch_size, y_dim+1, *x_shape]
            Reconstructed image. Values between 0,1 (i.e. after logistic).
            The first dimension corresponds to [labbeled] +
            [all possible values of y when unlabelled]

        z_suff_stat: torch.Tensor, size = [batch_size, y_dim+1, z_dim*2]
            Sufficient statistics of the latent sample {mu; logvar}.  The first
            dimension corresponds to [labbeled] + [all possible values of y when unlabelled]
        """
        y_hat = self.classifier(X)

        if y is None:
            y = torch.tensor([-1], device=y_hat.device).expand(y_hat.size(0))

        X_lab, X_unlab = split_labelled_unlabelled(X, y)
        y_lab, y_unlab = split_labelled_unlabelled(y, y)

        if (y != -1).sum() != 0:
            y_lab_onehot = torch.zeros_like(y_hat[:y_lab.size(0), ...]
                                            ).scatter_(1, y_lab.view(-1, 1), 1)
            rec_labs, z_sample_labs, z_suff_stat_labs = self._forward_labelled(X_lab, y_lab_onehot)
        else:
            rec_labs = z_sample_labs = z_suff_stat_labs = []

        if (y == -1).sum() != 0:
            # no copying
            y_unlab_onehot = torch.zeros_like(y_hat[:1, ...]).expand(y_unlab.size(0), self.y_dim)
            rec_unlabs, z_sample_unlabs, z_suff_stat_unlabs = self._forward_unlabelled(X_unlab, y_unlab_onehot)
        else:
            rec_unlabs = z_sample_unlabs = z_suff_stat_unlabs = []

        reconstruct = torch.stack(rec_labs + rec_unlabs, dim=1)
        z_sample = torch.stack(z_sample_labs + z_sample_unlabs, dim=1)
        z_suff_stat = torch.stack(z_suff_stat_labs + z_suff_stat_unlabs, dim=1)

        # inverts z_suff_stat, reconstruct because second output (z_sample) is
        # used for .transform while first output (y_hat) is used for .predict
        return y_hat, z_sample, reconstruct, z_suff_stat

    def _forward_labelled(self, X, y_onehot):
        z_suff_stat = self.encoder(X, y_onehot)
        z_sample = self.reparameterize(z_suff_stat)
        reconstruct = torch.sigmoid(self.decoder(torch.cat((z_sample, y_onehot), dim=1)))
        return [reconstruct], [z_sample], [z_suff_stat]

    def _forward_unlabelled(self, X, y_onehot):
        """Same as for labelled but marginalizes out labels => compute outputs for all y"""
        reconstruct_list, z_sample_list, z_suff_stat_list = [], [], []

        # expectation over all possible labels
        for l in range(self.y_dim):
            y_onehot.zero_()
            y_onehot[:, l] = 1

            reconstruct, z_sample, z_suff_stat = self._forward_labelled(X, y_onehot)
            reconstruct_list += reconstruct
            z_suff_stat_list += z_suff_stat
            z_sample_list += z_sample

        return reconstruct_list, z_sample_list, z_suff_stat_list

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
