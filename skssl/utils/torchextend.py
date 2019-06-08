import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.distributions.independent import Independent
from torch.distributions import Normal


def reduce(x, reduction="mean"):
    """Batch reduction of a tensor."""
    if reduction == "sum":
        x = x.sum()
    elif reduction == "mean":
        x = x.mean()
    elif reduction == "none":
        x = x
    else:
        raise ValueError("unkown reduction={}.".format(reduction))
    return x


def l1_loss(pred, target, reduction="mean"):
    """Computes the F1 loss with subgradient 0."""
    diff = pred - target
    loss = torch.abs(diff)
    loss = reduce(loss, reduction=reduction)
    return loss


def huber_loss(pred, target, delta=1e-3, reduction="mean"):
    """Computes the Huber loss."""
    diff = pred - target
    abs_diff = torch.abs(diff)
    loss = torch.where(abs_diff < delta,
                       0.5 * diff**2,
                       delta * (abs_diff - 0.5 * delta))
    loss = reduce(loss, reduction=reduction)
    return loss


def ReversedConv2d(in_filter, out_filter, *args, **kwargs):
    """Called the exact same way as Conv2d => with same in and out filter!"""
    return nn.ConvTranspose2d(out_filter, in_filter, *args, **kwargs)


def ReversedLinear(in_size, out_size, *args, **kwargs):
    """Called the exact same way as Linear => with same in and out dim!"""
    return nn.Linear(out_size, in_size, *args, **kwargs)


def identity(x):
    """simple identity function"""
    return x


def min_max_scale(tensor, min_val=0, max_val=1, dim=0):
    """Rescale value to be in a given range across dim."""
    tensor = tensor.float()
    std_tensor = (tensor - tensor.min(dim=dim, keepdim=True)[0]
                  ) / (tensor.max(dim=dim, keepdim=True)[0] - tensor.min(dim=dim, keepdim=True)[0])
    scaled_tensor = std_tensor * (max_val - min_val) + min_val
    return scaled_tensor


def MultivariateNormalDiag(loc, scale_diag):
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)


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


# TO-DO: use pytorch distributions
def kl_normal_loss(p_suff_stat, q_suff_stat=None):
    """
    Calculates the KL divergence between 2 normal distribution with with diagonal
    covariance for each example in a batch. `KL[P||Q].

    Parameters
    ----------
    p_suff_stat: torch.Tensor, size = [batch_size, 2*latent_dim]
        Mean and diagonal log variance of the first normal distribution.

    q_suff_stat: torch.Tensor, size = [batch_size, 2*latent_dim]
        Mean and diagonal log variance of the second normal distribution. If
        None, assumes standard normal gaussian.
    """
    p_mean, p_logvar = p_suff_stat.view(p_suff_stat.shape[0], -1, 2).unbind(-1)

    if q_suff_stat is None:
        q_mean = torch.zeros_like(p_mean)
        q_logvar = torch.zeros_like(p_logvar)
    else:
        q_mean, q_logvar = q_suff_stat.view(q_suff_stat.shape[0], -1, 2).unbind(-1)

    logvar_ratio = p_logvar - q_logvar

    t1 = (p_mean - q_mean).pow(2) / q_logvar.exp()

    kl = 0.5 * (t1 + logvar_ratio.exp() - 1 - logvar_ratio).sum(dim=-1)
    return kl


def reparameterize(mean_logvar, is_sample=True):
    """
    Samples from a normal distribution using the reparameterization trick.

    Parameters
    ----------
    mean_logvar: torch.Tensor, size = [batch_size, 2*latent_dim]
        Mean and diagonal log variance of the normal distribution.

    is_sample: bool, optional
        Whetehr to return a sample from the gaussian. If `False` returns the mean.
    """
    mean, logvar = mean_logvar.view(mean_logvar.shape[0], -1, 2).unbind(-1)
    std = torch.exp(0.5 * logvar)
    return reparameterize_meanstd(mean, std, is_sample=is_sample)


def reparameterize_meanstd(mean, std, is_sample=True):
    """
    Samples from a normal distribution using the reparameterization trick.

    Parameters
    ----------
    mean: torch.Tensor, size = [batch_size, latent_dim]
        Mean of the normal distribution

    std: torch.Tensor, size = [batch_size, latent_dim]
        Standard deviation of the normal distribution.

    is_sample: bool, optional
        Whetehr to return a sample from the gaussian. If `False` returns the mean.
    """
    if is_sample:
        eps = torch.randn_like(std)
        return mean + std * eps
    else:
        # Reconstruction mode
        return mean
