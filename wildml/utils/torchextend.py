import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.distributions.independent import Independent
from torch.distributions import Normal, Categorical, kl_divergence

from .initialization import weights_init


def min_jensen_shannon_div(p1, p2):
    M = Categorical(probs=(p1 + p2) / 2)
    return torch.min(kl_divergence(Categorical(probs=p1), M) +
                     kl_divergence(Categorical(probs=p2), M))


def jensen_shannon_div(p1, p2):
    p_avg = (p1 + p2) / 2
    mask = (p_avg != 0).float()
    # set to 0 p when M is 0 (because mean can only be 0 is vectors weree, but
    # this is not the case due to numerical issues)
    M = Categorical(probs=p_avg)
    return ((kl_divergence(Categorical(probs=p1 * mask), M) +
             kl_divergence(Categorical(probs=p2 * mask), M)) / 2)


def total_var(p1, p2):
    return (p1 - p2).abs().sum(-1) / 2


def hellinger_dist(p1, p2):
    return (p1.sqrt() - p2.sqrt()).pow(2).sum(-1).sqrt() / (2**0.5)


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
    if is_sample:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps
    else:
        # Reconstruction mode
        return mean


class ReturnInput(nn.Module):
    """Return the `which_input` input without any transformation."""

    def __init__(self, which_input):
        super().__init__()
        self.which_input = which_input

    def forward(self, *args):
        return args[self.which_input - 1]


def make_depth_sep_conv(Conv):
    """Make a convolution module depth separable."""
    class DepthSepConv(nn.Module):
        """Make a convolution depth separable.

        Parameters
        ----------
        in_channels : int
            Number of input channels.

        out_channels : int
            Number of output channels.

        kernel_size : int

        **kwargs :
            Additional arguments to `Conv`
        """

        def __init__(self, in_channels, out_channels, kernel_size,
                     bias=True, **kwargs):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.bias = bias
            self.depthwise = Conv(in_channels, in_channels, kernel_size,
                                  groups=in_channels, bias=bias, **kwargs)
            self.pointwise = Conv(in_channels, out_channels, 1, bias=bias)
            self.reset_parameters()

        # dirsty code to make it seem like a usual convolution
        @property
        def weight(self): return self.depthwise.weight

        @property
        def stride(self): return self.depthwise.stride

        @property
        def padding(self): return self.depthwise.padding

        @property
        def dilation(self): return self.depthwise.dilation

        def forward(self, x):
            out = self.depthwise(x)
            out = self.pointwise(out)
            return out

        def reset_parameters(self):
            weights_init(self)

    return DepthSepConv


def make_abs_conv(Conv):
    class AbsConv(Conv):
        def forward(self, input):
            return F.conv2d(input, self.weight.abs(), self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
    return AbsConv