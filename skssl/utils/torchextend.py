import torch
import torch.nn as nn
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
