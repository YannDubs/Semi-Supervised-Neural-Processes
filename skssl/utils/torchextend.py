import torch.nn as nn
from torch.distributions.independent import Independent
from torch.distributions import Normal


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
