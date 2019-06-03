import torch

import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import ClassifierMixin

from skssl.transformers.vae import VAE, VAELoss
from skssl.predefined import MLP
from skssl.utils.initialization import weights_init
from skssl.utils.helpers import split_labelled_unlabelled


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
        - DOesn't compute the loss due to p(y) because doesn't depend on param

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
        # reverts back to have (y_hat, reconstruct, z_sample, *) like for VAELoss
        inputs = inputs[:1] + inputs[2:0:-1] + inputs[2:]

        X_lab, X_unlab = split_labelled_unlabelled(X, y)
        # the first n_lab are the labelled ones and the rest are unlabelled
        inputs_lab, inputs_unlab = split_labelled_unlabelled(inputs, y, is_ordered=True)
        labelled_loss = unlabelled_loss = 0

        if (y != -1).sum() != 0:
            labelled_loss = self._labelled_loss(inputs_lab[1:], X_lab)
        if (y == -1).sum() != 0:
            unlabelled_loss = self._unlabelled_loss(inputs_unlab[1:], X_unlab, inputs_lab[0])

        # - log q(y|x)
        classifier_loss = self.criterion_y(inputs[0], y)

        return labelled_loss + unlabelled_loss + self.alpha * classifier_loss

    def _unlabelled_loss(self, vae_inputs, X, y_hat):
        """for unlabelled data: E_y [labelled loss] - H[q(y|x)]"""
        p_y_hat = F.softmax(y_hat, dim=-1)
        n_unlab, y_dim = y_hat.shape

        def get_vae_inputs(i):
            # unlabelled vae inputs are batch concatenated for all labels
            return tuple(inp[i * n_unlab:(i + 1) * n_unlab] for inp in vae_inputs)

        expected_lab_loss = sum(self._labelled_loss(get_vae_inputs(i), X, weight=p_y_hat[:, i])
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
class SSLVAE(VAE, ClassifierMixin):
    """Semisupervised VAE from [1] (M2 model). This is aclassifier, so the predicted
    label is the output of `predict` but `sample_decode` can still be used for generation.

    Parameters
    ----------
    Encoder : nn.Module
        Encoder module which maps x,y -> z. It should be callable with
        `encoder(x_shape, y_dim, n_out)`.

    Decoder : nn.Module
        Decoder module which maps z,y -> x. It should be callable with
        `decoder(x_shape, z_dim)`. No non linearities should be applied to the
        output.

    Classifier : nn.Module
        Classifier module which mapy X -> y. The last layer should be a softmax.

    x_shape : array-like
        Shape of a single example x.

    y_dim : int
        Number of classes.

    z_dim : int, optional
        Number of latent dimensions.

    Reference
    ---------
    [1] Kingma, D. P., Mohamed, S., Rezende, D. J., & Welling, M. (2014).
        Semi-supervised learning with deep generative models. In Advances in neural
        information processing systems (pp. 3581-3589).
    """

    def __init__(self, Encoder, Decoder, Classifier, x_shape, y_dim, z_dim=10):
        empty = lambda *args: None
        super().__init__(empty, empty, x_shape, z_dim)
        # initialize here as need y_dim
        self.encoder = Encoder(x_shape, y_dim, z_dim * 2)
        self.decoder = Decoder(x_shape, y_dim + z_dim)
        self.classifier = Classifier(x_shape, y_dim)
        self.y_dim = y_dim

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X, y=None):
        """
        Forward pass of model.

        Parameters
        ----------
        X : torch.Tensor, size = [*x_shape]
            Batch of data.

        y : torch.Tensor, size = [batch_size]
            Labels. -1 for unlabelled. `None` if all unlabelled.
        """
        y_hat = self.classifier(X)

        if y is None:
            y = torch.tensor([-1], device=y_hat.device).expand(y_hat.size(0))

        X_lab, X_unlab = split_labelled_unlabelled(X, y)
        y_lab, y_unlab = split_labelled_unlabelled(y, y)

        empty = torch.tensor([], dtype=y_hat.dtype, device=y_hat.device)
        reconstruct_lab = z_suff_stat_lab = z_sample_lab = empty
        reconstruct_unlab = z_suff_stat_unlab = z_sample_unlab = empty

        if (y != -1).sum() != 0:
            y_lab_onehot = torch.zeros_like(y_hat[:y_lab.size(0), ...]
                                            ).scatter_(1, y_lab.view(-1, 1), 1)
            reconstruct_lab, z_sample_lab, z_suff_stat_lab, = self._forward_labelled(X_lab, y_lab_onehot)

        if (y == -1).sum() != 0:
            # no copying
            y_unlab_onehot = torch.zeros_like(y_hat[:1, ...]).expand(y_unlab.size(0), self.y_dim)
            reconstruct_unlab, z_sample_unlab, z_suff_stat_unlab, = self._forward_unlabelled(X_unlab, y_unlab_onehot)

        reconstruct = torch.cat((reconstruct_lab, reconstruct_unlab), dim=0)
        z_suff_stat = torch.cat((z_suff_stat_lab, z_suff_stat_unlab), dim=0)
        z_sample = torch.cat((z_suff_stat_lab, z_suff_stat_unlab), dim=0)

        # inverts z_suff_stat, reconstruct because second is used for .transform
        return y_hat, z_sample, reconstruct, z_suff_stat

    def _forward_labelled(self, X, y_onehot):
        z_suff_stat = self.encoder(X, y_onehot)
        z_sample = self.reparameterize(z_suff_stat)
        reconstruct = torch.sigmoid(self.decoder(torch.cat((z_sample, y_onehot), dim=1)))
        return reconstruct, z_sample, z_suff_stat

    def _forward_unlabelled(self, X, y_onehot):
        """Same as for labelled but marginalizes out labels => compute outputs for all y"""
        reconstruct_list, z_sample_list, z_suff_stat_list = [], [], []

        for l in range(self.y_dim):
            y_onehot.zero_()
            y_onehot[:, l] = 1

            reconstruct, z_sample, z_suff_stat = self._forward_labelled(X, y_onehot)
            reconstruct_list.append(reconstruct)
            z_suff_stat_list.append(z_suff_stat)
            z_sample_list.append(z_sample)

        reconstruct = torch.cat(reconstruct_list, dim=0)
        z_suff_stat = torch.cat(z_suff_stat_list, dim=0)
        z_sample = torch.cat(z_sample_list, dim=0)

        return reconstruct, z_sample, z_suff_stat

    def sample_decode(self, z, y):
        """
        Returns a sample from the decoder.

        Parameters
        ----------
        z : torch.Tensor, size = [batch_size, latent_dim]
            Latent variable.

        y : torch.Tensor, size = [batch_size, y_dim]
            One hot encoded labels.
        """
        y = y.float()
        sample = torch.sigmoid(torch.cat((z, y), dim=1))
        return sample


# HELPERS
class SSLEncoder(nn.Module):
    """Ssl encoder from an encoder (i.e. takes y as input). The encoder is used
    to output z_tmp which will be concatenated with y and passed to a MLP to give z.

    Parameters
    ----------
    Encoder: nn.Module
        Encoder module which maps x -> z_tmp.

    x_shape: array-like
        Shape of a single example x.

    y_dim: int
        Number of classes.

    n_out: int, optional
        Size of ouput.

    tmp_out_dim: int, optional
        Temporary  dimension outputed by the encoder.
    """

    def __init__(self, Encoder, x_shape, y_dim, n_out, tmp_out_dim=128, **kwargs):
        super().__init__()
        self.encoder = Encoder(x_shape, tmp_out_dim, **kwargs)
        self.mixer = MLP(tmp_out_dim + y_dim, n_out, hidden_size=tmp_out_dim)

    def forward(self, x, y):
        tmp_z = self.encoder(x)
        z = self.mixer(torch.cat((tmp_z, y), dim=1))
        return z

    def reset_parameters(self):
        weights_init(self)


def make_ssl_encoder(encoder):
    """COnverts an encoder to an ssl encoder."""
    return lambda *args, **kwargs: SSLEncoder(encoder, *args, **kwargs)
