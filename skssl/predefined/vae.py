import torch
import torch.nn as nn
from torch.nn import functional as F

from sklearn.base import TransformerMixin

from skssl.utils.initialization import weights_init


# LOSS
class VAELoss(nn.Module):
    """
    Compute the VAE or Beta-VAE loss as in [1].

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence. If 1, corresponds to standard VAE.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    distribution : {"bernoulli", "gaussian", "laplace"}, optional
        Distribution of the likelihood for each feature. See `reconstruction_loss`
        for more details.

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """

    def __init__(self, beta=1, distribution="laplace"):
        super().__init__(self)
        self.beta = beta
        self.distribution = distribution

    def forward(self, inputs, targets):
        """Compute the VAE loss.

        Parameters
        ----------
        inputs : tuple
            Tuple of (reconstruct, z_dist, *). This can directly take the output
            of VAE.

        target : torch.Tensor, size = [batch_size, *x_shape]
            Targets.
        """
        reconstruct, z_dist = inputs[:2]
        rec_loss = reconstruction_loss(targets, reconstruct, distribution=z_dist)
        kl_loss = kl_normal_loss(*z_dist)
        loss = rec_loss + self.beta * kl_loss
        return loss


def reconstruction_loss(data, recon_data, distribution="bernoulli"):
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
    loss : torch.Tensor
        Per example cross entropy (i.e. normalized per batch but not per features)
    """

    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
    elif distribution == "gaussian":
        loss = F.mse_loss(recon_data, data, reduction="sum")
    elif distribution == "laplace":
        loss = F.l1_loss(recon_data, data, reduction="sum")
        loss = loss * (loss != 0)  # masking to avoid nan
    else:
        raise ValueError("Unkown distribution: {}".format(distribution))

    loss = loss / recon_data.size(0)

    return loss


def kl_normal_loss(mean, logvar):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor, size = [batch_size, z_dim]
        Mean of the normal distribution.

    logvar : torch.Tensor, size = [batch_size, z_dim]
        Diagonal log variance of the normal distribution.
    """
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()
    return total_kl


# MODEL
class VAE(nn.Module, TransformerMixin):
    """Unsupervised VAE. This is a transformer as it is primarily
    used for representation learning, so the mean latent is the output of `predict`
    but `sample_decode` can still be used for generation.

    Parameters
    ----------
    encoder : nn.Module
        Encoder module which maps x -> z. It should be callable with
        `encoder(x_shape, n_out)`.

    decoder : nn.Module
        Decoder module which maps z -> x. It should be callable with
        `decoder(x_shape, n_out)`. No non linearities should be applied to the
        output.

    x_shape : tuple of ints
        Shape of a single example x.

    z_dim : int, optional
        Number of latent dimensions.
    """

    def __init__(self, encoder, decoder, x_shape, z_dim=10):
        self.encoder = encoder(x_shape, z_dim * 2)
        self.decoder = decoder(x_shape, z_dim)
        self.z_dim = z_dim
        self.x_shape = x_shape

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

    def forward(self, X, y):
        """
        Forward pass of model.

        Parameters
        ----------
        X : torch.Tensor, size = [*x_shape]
            Batch of data.

        y : torch.Tensor, size = [batch_size, y_dim]
            One hot encoded labels.
        """
        z_dist = self.encoder(X)
        mean, logvar = z_dist.view(-1, self.z_dim, 2).unbind(-1)
        z_sample = self.reparameterize(mean, logvar)
        reconstruct = torch.sigmoid(self.decoder(z_sample))
        return reconstruct, (mean, logvar), z_sample

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor, size = [batch_size, latent_dim]
            Mean of the normal distribution.

        logvar : torch.Tensor, size = [batch_size, latent_dim]
            Diagonal log variance of the normal distribution.
        """
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
