import torch
import torch.nn as nn
from torch.nn import functional as F

from sklearn.base import TransformerMixin

from skssl.predefined import WideResNet, ReversedSimpleCNN
from skssl.utils.initialization import weights_init
from skssl.utils.torchextend import l1_loss, huber_loss

__all__ = ["VAELoss", "VAE"]


# LOSS
class VAELoss(nn.Module):
    """
    Compute the VAE or Beta-VAE loss as in [1].

    Parameters
    ----------
    get_beta : callable, optional
        Function which returns the weight of the kl divergence given `is_training`
        . Returning a constant 1 corresponds to standard VAE.

    distribution : {"bernoulli", "gaussian", "laplace"}, optional
        Distribution of the likelihood for each feature. See `reconstruction_loss`
        for more details.

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """

    def __init__(self, get_beta=lambda _: 1, distribution="laplace"):
        super().__init__()
        self.get_beta = get_beta
        self.distribution = distribution

    def forward(self, inputs, y=None, X=None, weight=None):
        """Compute the VAE loss averaged over the batch.

        Parameters
        ----------
        inputs : tuple
            Tuple of (reconstruct, z_sample, z_suff_stat, *). This can directly
            take the output of VAE.

        y : None
            Placeholder.

        X : torch.Tensor or dict containing X, size = [batch_size, *x_shape]
            Training data corresponding to the targets.

        weight : torch.Tensor, size = [batch_size,]
            Weight of every example. If None, every example is weighted by 1.
        """
        X = X["X"] if isinstance(X, dict) else X
        assert X is not None
        reconstruct, z_sample, z_suff_stat = inputs[:3]
        rec_loss = reconstruction_loss(X, reconstruct, distribution=self.distribution)
        kl_loss = kl_normal_loss(z_suff_stat)
        loss = rec_loss + self.get_beta(self.training) * kl_loss

        if weight is not None:
            loss = loss * weight

        return loss.mean(dim=0)


def reconstruction_loss(data, recon_data, distribution="laplace"):
    """
    Calculates the per image reconstruction loss for a batch of data. I.e. negative
    log likelihood.

    Parameters
    ----------
    data : torch.Tensor, size = [batch_size, *]
        Input data (e.g. batch of images).

    recon_data : torch.Tensor, size = [batch_size, *]
        Reconstructed data.

    distribution : {"bernoulli", "gaussian", "laplace"}, optional
        Distribution of the likelihood for each feature. Implicitely defines the
        loss. Bernoulli corresponds to a binary cross entropy (bse) loss and is the
        most commonly used for features in [0,1] (e.g. normalized images). It has
        the issue that it doesn't penalize the same way (0.1,0.2) and (0.4,0.5),
        which might not be optimal. Gaussian distribution corresponds to MSE,
        and is sometimes used, but hard to train because it ends up focusing only
        a few features that are very wrong. Laplace distribution corresponds to
        L1 solves partially the issue of MSE.

    Returns
    -------
    loss : torch.Tensor, size = [batch_size,]
        Loss for each example.
    """
    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="none")
    elif distribution == "gaussian":
        loss = F.mse_loss(recon_data, data, reduction="none")
    elif distribution == "laplace":
        loss = l1_loss(recon_data, data, reduction="none")
    else:
        raise ValueError("Unkown distribution: {}".format(distribution))

    batch_size = recon_data.size(0)
    loss = loss.view(batch_size, -1).sum(dim=1)

    return loss


def kl_normal_loss(z_suff_stat):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution for each example
    in a batch.

    Parameters
    ----------
    z_suff_stat : torch.Tensor, size = [batch_size, 2*latent_dim]
        Mean and diagonal log variance of the normal distribution.
    """
    mean, logvar = z_suff_stat.view(z_suff_stat.shape[0], -1, 2).unbind(-1)
    kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).sum(dim=1)
    return kl


# MODEL
class VAE(nn.Module, TransformerMixin):
    """Unsupervised VAE. This is a transformer as it is primarily
    used for representation learning, so the mean latent is the output of `predict`
    but `sample_decode` can still be used for generation.

    Parameters
    ----------
    Encoder : nn.Module
        Encoder module which maps x -> z_suff_stat. It should be callable with
        `encoder(x_shape, n_out)`.

    Decoder : nn.Module
        Decoder module which maps z -> x. It should be callable with
        `decoder(z_dim, x_shape)`. No non linearities should be applied to the
        output.

    x_shape : tuple of ints
        Shape of a single example x.

    z_dim : int, optional
        Number of latent dimensions.
    """

    def __init__(self, x_shape,
                 Encoder=WideResNet,
                 Decoder=ReversedSimpleCNN,
                 z_dim=10):
        super().__init__()
        self.x_shape = x_shape
        self.encoder = Encoder(x_shape, z_dim * 2)
        self.decoder = Decoder(z_dim, x_shape)
        self.z_dim = z_dim

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):
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
        reconstruct: torch.Tensor, size = [batch_size, *x_shape]
            Reconstructed image. Values between 0,1 (i.e. after logistic).

        z_sample: torch.Tensor, size = [batch_size, z_dim]
            Latent sample.

        z_suff_stat: torch.Tensor, size = [batch_size, z_dim*2]
            Sufficient statistics of the latent sample {mu; logvar}.
        """
        z_suff_stat = self.encoder(X)
        z_sample = self.reparameterize(z_suff_stat)
        reconstruct = torch.sigmoid(self.decoder(z_sample))
        return reconstruct, z_sample, z_suff_stat

    def reparameterize(self, z_suff_stat):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        z_suff_stat : torch.Tensor, size = [batch_size, 2*latent_dim]
            Mean and diagonal log variance of the normal distribution.
        """
        mean, logvar = z_suff_stat.view(z_suff_stat.shape[0], -1, 2).unbind(-1)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def sample_decode(self, z):
        """
        Returns a sample from the decoder.

        Parameters
        ----------
        z : torch.Tensor, size = [batch_size, latent_dim]
            Latent variable.
        """
        sample = torch.sigmoid(self.decoder(z))
        return sample
