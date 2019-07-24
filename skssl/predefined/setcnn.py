import math
import warnings
import itertools
from operator import xor

import torch
import torch.nn as nn
from torch.nn import functional as F

from .mlp import MLP
from skssl.utils.helpers import mask_and_apply
from skssl.utils.initialization import weights_init, init_param_


__all__ = ["SetConv", "SparseSetConv", "MlpRBF", "GaussianRBF", "BatchSparseSetConv"]


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

    def __init__(self, x_dim, is_abs_dist=True, window_size=0.25,
                 is_sparse=False, **kwargs):
        super().__init__()
        self.is_abs_dist = is_abs_dist
        self.window_size = window_size
        self.is_sparse = is_sparse
        self.mlp = MLP(x_dim, 1, n_hidden_layers=3, hidden_size=16)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, diff):
        abs_diff = diff.abs()

        # select only points with distance (first dim) less than window_size
        # (for extrapolation + speed)
        mask = (abs_diff[..., :1] < self.window_size)

        if self.is_abs_dist:
            diff = abs_diff

        # sparse operation (apply only on mask) => 2-3x speedup
        weight = mask_and_apply(diff, mask, lambda x: self.mlp(x).abs())
        weight = weight * mask.float()  # set to 0 points that are further than windo

        if self.is_sparse:
            weight = weight * abs_diff[..., 1:]

        density = weight.sum(dim=-2, keepdim=True)
        out = weight / (density + 1e-5)  # don't divide by 0

        return out, density.squeeze(-2)


class SetConv(nn.Module):
    """Applies a convolution over a set of inputs, i.e. generalizes `nn._ConvNd`
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
        Number of output channels. If `None` doesn't use a linear layer for resizing.

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
        self.out_channels = out_channels

        self.in_channels = in_channels + is_concat_density
        self.radial_basis_func = RadialBasisFunc(x_dim, is_vanilla=is_vanilla,
                                                 max_dist=max_dist, **kwargs)

        self.is_concat_density = is_concat_density
        self.is_vanilla = is_vanilla

        if self.is_concat_density and not self.is_vanilla:
            self.density_transform = nn.Linear(1, 1)

        if self.out_channels is not None:
            self.resizer = nn.Linear(self.in_channels, self.out_channels)
        else:
            self.resizer = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, keys, queries, values):
        """
        Compute the set convolution between {key, value} and {querry}.

        Parameters
        ----------
        keys : torch.Tensor, size=[batch_size, n_keys, x_dim]
        queries : torch.Tensor, size=[batch_size, n_queries, x_dim]
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


# try sparse cnn with one hot encoding to use a single MLP and be able to make
# everything batch computable
class SparseSetConv(nn.Module):
    """Applies a convolution over a set of inputs with sparse channels. I.e.
    extends `SetConv` for inputs with sparse channels.

    Parameters
    ----------
    x_dim : int
        Dimensionality of input. This should be one less than the actual input
        dimension, as the first dimension will represent the channel. Currently
        onyl works for `x_dim==1`.

    in_channels : int
        Number of input channels. When `is_sparse_keys` this is the number of sparse
        channels.

    out_channels : int
        Number of output channels.

    is_diff_channels : bool, optional
        Whether the channels should all be treated with their own `SetConv`, this
        is computationally more expensive but should be better if all chhanels
        represent very different things.

    is_sparse_keys : bool, optional
        Whether the keys will be sparse. CUrrently only works with sparse keys

    is_sparse_queries : bool, optional
        Whether the queries will be sparse. In this case the output will not
        be sparse (always outputs every channel).

    kwargs :
        Additional arguments to `SetConv`.
    """

    def __init__(self, x_dim, in_channels, out_channels,
                 is_diff_channels=False, is_sparse_keys=True,
                 is_sparse_queries=False, **kwargs):

        assert x_dim == 1, "Currently only works with x_dim=1"
        assert xor(is_sparse_keys, is_sparse_queries)

        super().__init__()
        self.in_channels = in_channels
        self.is_diff_channels = is_diff_channels
        self.is_sparse_keys = is_sparse_keys
        self.is_sparse_queries = is_sparse_queries
        self.x_dim = x_dim

        if self.is_diff_channels and self.is_sparse_keys:
            self.setconvs = nn.ModuleList([SetConv(x_dim, 1, None,
                                                   is_concat_density=True, **kwargs)
                                           for _ in range(self.in_channels)])
        else:
            self.setconvs = SetConv(x_dim, 1, None, is_concat_density=True, **kwargs)

        resizer_input_dim = self.in_channels * 2 if self.is_sparse_keys else self.in_channels + 1
        self.resizer = nn.Linear(resizer_input_dim, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def _sparse_to_list(self, sparse_tensor, *other_tensors, is_sparse=True):
        """Converts a sparse tensor to a list of tensors based on the channel dimension."""
        # all tensors to split, without channel dim
        if is_sparse:
            all_tensors = [sparse_tensor[..., 1:]] + list(other_tensors)

            channels = sparse_tensor[..., 0]
            list_of_tensors = [[None] * self.in_channels
                               for _ in range(len(all_tensors))]
            for c in range(self.in_channels):
                idcs = channels.int() == c
                for i, t in enumerate(all_tensors):
                    list_of_tensors[i][c] = t[idcs].unsqueeze(0)

        else:
            all_tensors = [sparse_tensor] + list(other_tensors)

            if sparse_tensor.size(-1) == self.x_dim:
                list_of_tensors = [itertools.repeat(t, self.in_channels)
                                   for t in all_tensors]
            elif sparse_tensor.size(-1) == self.in_channels:
                list_of_tensors = [t.unbind(-1) for t in all_tensors]
            else:
                raise ValueError("Bad input size: {}. x_dim={}, in_channels={}.".format(sparse_tensor.shape, self.x_dim, self.in_channels))

        if len(list_of_tensors) == 1:
            list_of_tensors = list_of_tensors[0]

        return list_of_tensors

    def _forward_single_batch(self, keys, queries, values):
        """
        Compute the set convolution for a single batch
        """
        assert self.is_sparse_keys
        list_keys, list_values = self._sparse_to_list(keys, values,
                                                      is_sparse=self.is_sparse_keys)
        list_queries = self._sparse_to_list(queries, is_sparse=False)
        # list_queries = itertools.repeat(queries[..., 1:], self.in_channels)

        targets = [None] * self.in_channels
        for i, (keys, queries, values) in enumerate(zip(list_keys, list_queries, list_values)):
            if keys.numel() == 0:
                # if empty then give vectors of all zeros (density is 0 so network
                # can know that no data)
                targets[i] = torch.zeros_like(queries.expand(*queries.shape[:-1], 2))
            else:
                set_conv = self.setconvs[i] if self.is_diff_channels else self.setconvs
                targets[i] = set_conv(keys, queries, values)

        targets = torch.cat(targets, dim=-1)

        return targets

    def forward(self, keys, queries, values):
        """
        Compute the set convolution between {key, value} and {querry}.

        Parameters
        ----------
        keys : torch.Tensor, size=[batch_size, n_keys, x_dim+is_sparse_keys]
        queries : torch.Tensor, size=[batch_size, n_queries, x_dim+is_sparse_queries]
        values : torch.Tensor, size=[batch_size, n_keys, 1]

        Return
        ------
        targets : torch.Tensor, size=[batch_size, n_queries, self.out_channels]
        """
        if self.is_sparse_keys:
            batch_size = keys.size(0)
            # split all batches (because different size, then run setcnn and stack back)
            targets = torch.cat([self._forward_single_batch(keys[b:b + 1, ...],
                                                            queries[b:b + 1, ...],
                                                            values[b:b + 1, ...])
                                 for b in range(batch_size)], dim=0)
        elif self.is_sparse_queries:
            # drop sparse of queries
            targets = self.setconvs(keys, queries[..., 1:], values)

        targets = self.resizer(targets)

        return targets


class BatchSparseSetConv(nn.Module):
    """Applies a convolution over a set of inputs with sparse channels. I.e.
    extends `SetConv` for inputs with sparse channels.

    Parameters
    ----------
    x_dim : int
        Dimensionality of input. This should be one less than the actual input
        dimension, as the first dimension will represent the channel. Currently
        onyl works for `x_dim==1`.

    in_channels : int
        Number of input channels. When `is_sparse_keys` this is the number of sparse
        channels.

    out_channels : int
        Number of output channels.

    is_diff_channels : bool, optional
        Whether the channels should all be treated with their own `SetConv`, this
        is computationally more expensive but should be better if all chhanels
        represent very different things.

    is_sparse_keys : bool, optional
        Whether the keys will be sparse. CUrrently only works with sparse keys

    is_sparse_queries : bool, optional
        Whether the queries will be sparse. In this case the output will not
        be sparse (always outputs every channel).

    kwargs :
        Additional arguments to `SetConv`.
    """

    def __init__(self, x_dim, in_channels, out_channels, **kwargs):

        assert x_dim == 1, "Currently only works with x_dim=1"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.x_dim = x_dim

        self.radial_basis_func = MlpRBF(self.in_channels + self.x_dim, is_sparse=True)
        self.density_transform = nn.Linear(1, 1)

        self.resizer = nn.Linear(self.in_channels * 2, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, keys, queries, values):
        """
        Compute the set convolution between {key, value} and {querry}.

        Parameters
        ----------
        keys : torch.Tensor, size=[batch_size, n_keys, x_dim]
        queries : torch.Tensor, size=[batch_size, n_queries, x_dim]
        values : torch.Tensor, size=[batch_size, n_keys, in_channels]

        Return
        ------
        targets : torch.Tensor, size=[batch_size, n_queries, out_channels]
        """

        channels = keys[..., 0]
        keys = keys[..., 1:]

        # prepares for broadcasted computations
        keys = keys.unsqueeze(1)
        queries = queries.unsqueeze(2)
        values = values.unsqueeze(1)

        # weight size = [batch_size, n_queries, n_keys, in_channels]
        # density size = [batch_size, n_queries, in_channels]
        diff = keys - queries
        one_hot = F.one_hot(channels.long(), num_classes=self.in_channels).float().unsqueeze(1)
        one_hot = one_hot.expand(*diff.shape[:-1], -1)
        weight, density = self.radial_basis_func(torch.cat([diff, one_hot], dim=-1))

        # size = [batch_size, n_queries, value_size]
        values = one_hot * values
        targets = (weight * values).sum(dim=2)

        density = torch.sigmoid(self.density_transform(density.unsqueeze(-1) * 0.1 - 1)
                                ).squeeze(-1)
        # don't normalize the density channel
        targets = torch.cat([targets, density], dim=-1)

        targets = self.resizer(targets)

        return targets
