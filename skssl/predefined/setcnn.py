import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from neuralproc.predefined import MLP
from .initialization import weights_init, init_param_
from skssl.utils.helpers import mask_and_apply

__all__ = ["SetConv", "MlpRBF", "GaussianRBF"]


class Diff2Dist(nn.Module):
    """Compute the (dimensionwise weighted) L2 distance.

    Parameters
    ----------
    x_dim : int
        Dimensionality of input.

    is_weight_dim :
        Whether to weight the the different dimensions.
    """

    def __init__(self, x_dim, is_weight_dim=True):
        super().__init__()
        self.is_weight_dim = is_weight_dim and x_dim != 1

        if self.is_weight_dim:
            self.weighter = nn.Linear(x_dim, x_dim)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)
        if self.is_weight_dim:
            # center the initialization to 1 => by default all weights close to 1
            init_param_(self.weighter.weight, shift=1)

    def forward(self, diff):

        if self.is_weight_dim:
            diff = self.weighter(diff)

        dist = torch.norm(diff, p=2, dim=-1, keepdim=True)
        return dist


class GaussianRBF(nn.Module):
    """Gaussian radial basis function.

    Parameters
    ----------
    x_dim : int
        Dimensionality of input.

    is_normalize : bool, optional
        Whether weights should sum to 1 (using softmax). If not weights will not
        be normalized.

    is_vanilla : bool, optional
        Whether to force the use of the vanilla setcnn.

    max_dist : float, optional
        Max distance between the closest query and target, used for intiialisation.

    max_dist_weight : float, optional
        Weight that should be given to a maximum distance. Note that min_dist_weight
        is 1, so can also be seen as a ratio.

    kwargs :
        Additional arguents to `Diff2Dist`.
    """

    def __init__(self, x_dim, is_normalize=True, is_vanilla=False, max_dist=1 / 256,
                 max_dist_weight=0.5, **kwargs):
        super().__init__()

        self.max_dist = max_dist
        self.max_dist_weight = max_dist_weight
        self.is_normalize = is_normalize and not is_vanilla
        self.length_scale_param = nn.Parameter(torch.tensor([0.]))
        self.diff2dist = Diff2Dist(x_dim, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)
        # set the parameter depending on the weight to give to a maxmum distance
        # query. i.e. exp(- (max_dist / sigma).pow(2)) = max_dist_weight
        # => sigma = max_dist / sqrt(- log(max_dist_weight))
        max_dist_sigma = self.max_dist / math.sqrt(- math.log(self.max_dist_weight))
        # inverse_softplus : log(exp(y) - 1)
        max_dist_param = math.log(math.exp(max_dist_sigma) - 1)
        self.length_scale_param = nn.Parameter(torch.tensor([max_dist_param]))

    def forward(self, diff):
        dist = self.diff2dist(diff)

        length_scale_param = self.length_scale_param
        # compute exponent making sure no division by 0
        sigma = 1e-5 + F.softplus(length_scale_param)
        inp = - (dist / sigma).pow(2)

        if self.is_normalize:
            # don't use fraction because numerical instability => softmax tricks
            out = torch.softmax(inp, dim=-2)
            density = torch.exp(inp).sum(dim=-2)
        else:
            out = torch.exp(inp)
            density = out.sum(dim=-2)

        return out, density


class MlpRBF(nn.Module):
    """Gaussian radial basis function.

    Parameters
    ----------
    x_dim : int
        Dimensionality of input.

    is_add_sin : bool, optional
        Whether to add a sinusoidal feature.

    is_abs_dist : bool, optional
        Whether to force the kernel to be sign invariant.

    window_size : int, bool
        Maximimum distance to consider

    kwargs :
        Placeholder
    """

    def __init__(self, x_dim, is_abs_dist=True, window_size=0.25, **kwargs):
        super().__init__()
        self.is_abs_dist = is_abs_dist
        self.window_size = window_size
        self.mlp = MLP(x_dim, 1, n_hidden_layers=3, hidden_size=16)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, diff):
        abs_diff = diff.abs()

        # select only points with distance less than window_size (for extrapolation + speed)
        mask = (abs_diff < self.window_size)

        if self.is_abs_dist:
            diff = abs_diff

        # sparse operation (apply only on mask) => 2-3x speedup
        weight = mask_and_apply(diff, mask,
                                lambda x: self.mlp(x.unsqueeze(1)).abs().squeeze())
        weight = weight * mask.float()  # set to 0 points that are further than windo

        density = weight.sum(dim=-2, keepdim=True)
        out = weight / (density + 1e-5)  # don't divide by 0

        return out, density.squeeze(-1)


class SetConv(nn.Module):
    """Applies a 1D convolution over a set of inputs, i.e. generalizes `nn._ConvNd`
    to non uniform.

    Note
    ----
    Corresponds to an attention mechanism if `RadialFunc=GaussianRBF` and
    `is_normalize` (corresponds to a weighted distance attention), but differs
    from usual attention for general `RadialFunc`.

    Parameters
    ----------
    x_dim : int
        Dimensionality of input.

    in_channels : int
        Number of input channels.

    out_channels : int
        Number of output channels.

    RadialBasisFunc : callable, optional
        Function which returns the "weight" of each points as a function of their
        distance (i.e. for usual CNN that would be the filter).

    is_concat_density : bool, optional
        Whether to concatenate a density channel. Will concatenate the density
        transformed by logistic function with learned temperature (making sure
        that sample invariant).

    is_vanilla : bool, optional
        Whether to force the use of the vanilla setcnn.

    max_dist : float, optional
            Max distance between the closest query and target, used for intiialisation.

    kwargs :
        Additional arguments to `RadialBasisFunc`.
    """

    def __init__(self, x_dim, in_channels, out_channels,
                 RadialBasisFunc=MlpRBF,
                 is_concat_density=True,
                 is_vanilla=False,
                 max_dist=1 / 256,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels + is_concat_density
        self.out_channels = out_channels
        self.radial_basis_func = RadialBasisFunc(x_dim, is_vanilla=is_vanilla,
                                                 max_dist=max_dist, **kwargs)
        self.is_concat_density = is_concat_density
        self.is_vanilla = is_vanilla

        if self.is_concat_density and not self.is_vanilla:
            self.density_transform = nn.Linear(1, 1)

        self.resizer = nn.Linear(self.in_channels, self.out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, keys, queries, values):
        """
        Compute the set convolution between {key, value} and {querry}.

        Parameters
        ----------
        keys : torch.Tensor, size=[batch_size, n_keys, kq_size]
        queries : torch.Tensor, size=[batch_size, n_queries, kq_size]
        values : torch.Tensor, size=[batch_size, n_keys, in_channels]

        Return
        ------
        targets : torch.Tensor, size=[batch_size, n_queries, value_size]
        """
        # prepares for broadcasted computations
        keys = keys.unsqueeze(1)
        queries = queries.unsqueeze(2)
        values = values.unsqueeze(1)

        # weight size = [batch_size, n_queries, n_keys, 1]
        # density size = [batch_size, n_queries, 1]
        weight, density = self.radial_basis_func(keys - queries)

        # size = [batch_size, n_queries, value_size]
        targets = (weight * values).sum(dim=2)

        if self.is_vanilla:
            # constant division that works well in practive
            targets = targets / 50
            density = density / 50
            targets = self.resizer(torch.cat([targets, density], dim=-1))
            return torch.sigmoid(targets)

        if self.is_concat_density:
            # same as using a smaller initialization to be close to 0.5 at start
            # and remove 1 because if nto always positive => will saturate sigmoid
            # detching because if has linear and sigma to change then high varince
            density = torch.sigmoid(self.density_transform(density * 0.1 - 1))
            # don't normalize the density channel
            targets = torch.cat([targets, density], dim=-1)

        targets = self.resizer(targets)

        return targets  # , density
