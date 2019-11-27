import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from essl.utils.initialization import weights_init, init_param_
from essl.utils.helpers import backward_pdb, mask_and_apply, ProbabilityConverter

from .mlp import MLP

__all__ = ["SetConv", "MlpRBF", "GaussianRBF"]


class Diff2Dist(nn.Module):
    """Compute the (dimensionwise weighted) L2 distance.

    Parameters
    ----------
    x_dim : int
        Number of spatio-temporal input dimensions.

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

    max_dist : float, optional
        Max distance between the closest query and target, used for intialisation.

    max_dist_weight : float, optional
        Weight that should be given to a maximum distance. Note that min_dist_weight
        is 1, so can also be seen as a ratio.

    kwargs :
        Additional arguents to `Diff2Dist`.
    """

    def __init__(self, x_dim, max_dist=1 / 256, max_dist_weight=0.5, **kwargs):
        super().__init__()

        self.max_dist = max_dist
        self.max_dist_weight = max_dist_weight
        self.length_scale_param = nn.Parameter(torch.tensor([0.0]))
        self.diff2dist = Diff2Dist(x_dim, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)
        # set the parameter depending on the weight to give to a maxmum distance
        # query. i.e. exp(- (max_dist / sigma).pow(2)) = max_dist_weight
        # => sigma = max_dist / sqrt(- log(max_dist_weight))
        max_dist_sigma = self.max_dist / math.sqrt(-math.log(self.max_dist_weight))
        # inverse_softplus : log(exp(y) - 1)
        max_dist_param = math.log(math.exp(max_dist_sigma) - 1)
        self.length_scale_param = nn.Parameter(torch.tensor([max_dist_param]))

    def forward(self, diff):
        dist = self.diff2dist(diff)

        # compute exponent making sure no division by 0
        sigma = 1e-5 + F.softplus(self.length_scale_param)

        inp = -(dist / sigma).pow(2)
        out = torch.softmax(inp, dim=-2)  # numerically stable normalization
        density = torch.exp(inp).sum(dim=-2)

        return out, density


class MlpRBF(nn.Module):
    """Gaussian radial basis function.

    Parameters
    ----------
    x_dim : int
        Number of spatio-temporal input dimensions.

    is_abs_dist : bool, optional
        Whether to force the kernel to be symmetric around 0.

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
        mask = abs_diff < self.window_size

        if self.is_abs_dist:
            diff = abs_diff

        # sparse operation (apply only on mask) => 2-3x speedup
        weight = mask_and_apply(diff, mask, lambda x: self.mlp(x.unsqueeze(1)).abs().squeeze())
        weight = weight * mask.float()  # set to 0 points that are further than windo

        density = weight.sum(dim=-2, keepdim=True)
        out = weight / (density + 1e-5)  # don't divide by 0

        return out, density.squeeze(-1)


class SetConv(nn.Module):
    """Applies a convolution over a set of inputs, i.e. generalizes `nn._ConvNd`
    to non uniformly sampled samples [jonathan].

    Parameters
    ----------
    x_dim : int
        Number of spatio-temporal dimensions of input.

    in_channels : int
        Number of input channels.

    out_channels : int
        Number of output channels.

    RadialBasisFunc : callable, optional
        Function which returns the "weight" of each points as a function of their
        distance (i.e. for usual CNN that would be the filter).

    kwargs :
        Additional arguments to `RadialBasisFunc`.

    References
    ----------
    [jonathan]
    """

    def __init__(self, x_dim, in_channels, out_channels, RadialBasisFunc=GaussianRBF, **kwargs):
        super().__init__()
        self.radial_basis_func = RadialBasisFunc(x_dim, **kwargs)
        self.resizer = nn.Linear(in_channels * 2, out_channels)
        self.density_to_conf = ProbabilityConverter(
            is_train_temperature=True,
            is_train_bias=True,
            trainable_dim=in_channels,
            # higher density => higher conf
            temperature_transformer=F.softplus,
        )
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, keys, queries, values):
        """
        Compute the set convolution between {key, value} and {querry}.

        TODO
        ----
        - should sort the keys and queries to not compute differences if outside
        of given receptive field (large memory savings).

        Parameters
        ----------
        keys : torch.Tensor, size=[batch_size, n_keys, kq_size]
        queries : torch.Tensor, size=[batch_size, n_queries, kq_size]
        values : torch.Tensor, size=[batch_size, n_keys, in_channels]

        Return
        ------
        targets : torch.Tensor, size=[batch_size, n_queries, out_channels]
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

        # initial density could be very large => make sure not saturating sigmoid (*0.1)
        confidence = self.density_to_conf(density * 0.1)
        # don't concatenate density but a bounded version ("confidence") =>
        # doesn't break under high density
        targets = torch.cat([targets, confidence], dim=-1)

        return self.resizer(targets)
