import random
import functools

import torch
import numpy as np

from skssl.utils.helpers import ratio_to_int, prod


def random_masker(batch_size, mask_shape,
                  min_nnz=.1,
                  max_nnz=.5,
                  is_batch_repeat=False):
    """
    Return a random context mask. The number of non zero values will be in
    [min_nnz, max_nnz]. If min_perc, max_perc are smaller than 1 these represent
    a percentage of points. If `is_batch_repeat` use the same mask for all elements
    in the batch.
    """
    n_possible_points = prod(mask_shape)
    min_nnz = ratio_to_int(min_nnz, n_possible_points)
    max_nnz = ratio_to_int(max_nnz, n_possible_points)
    n_nnz = random.randint(min_nnz, max_nnz)

    if is_batch_repeat:
        mask = torch.zeros(n_possible_points).byte()
        mask[torch.randperm(n_possible_points)[:n_nnz]] = 1
        mask = mask.view(*mask_shape).contiguous()
        mask = mask.unsqueeze(0).expand(batch_size, *mask_shape)
    else:
        mask = np.zeros((batch_size, n_possible_points), dtype=np.uint8)
        mask[:, :n_nnz] = 1
        indep_shuffle(mask, -1)
        mask = torch.from_numpy(mask)
        mask = mask.view(batch_size, *mask_shape).contiguous()

    return mask


def and_masks(*masks):
    """Composes tuple of masks by an and operation."""
    mask = functools.reduce(lambda a, b: a & b, masks)
    return mask


def or_masks(*masks):
    """Composes tuple of masks by an or operation."""
    mask = functools.reduce(lambda a, b: a | b, masks)
    return mask


def half_masker(batch_size, mask_shape, dim=0):
    """Return a mask which masks the top half features of `dim`."""
    mask = torch.zeros((batch_size, *mask_shape)).byte()
    slcs = [slice(None)] * (len(mask_shape))
    slcs[dim] = slice(0, mask_shape[dim] // 2)
    mask[[slice(None)] + slcs] = 1
    return mask


def no_masker(batch_size, mask_shape):
    """Return a mask of all 1."""
    mask = torch.ones((batch_size, *mask_shape)).byte()
    return mask


def context_target_split(X, Y,
                         range_cntxts=(.1, .5),
                         range_extra_trgts=(.1, .5),
                         is_all_trgts=False):
    """Given inputs X and their value y, return random subsets of points for
    context and target.

    Notes
    -----
    - modified from: github.com/EmilienDupont/neural-processes
    - context will be part of targets

    Parameters
    ----------
    X : torch.Tensor, size = [batch_size, num_points, x_dim]
        Position features. Values should always be in [-1,1].

    Y : torch.Tensor, size = [batch_size, num_points, y_dim]
        Targets.

    range_cntxts : tuple of int or floats, optional
        Range for the number of context points (min_range, max_range) if values are
        smaller than 1 these represent a percentage of points.

    range_extra_trgts : tuple of int or floats, optional
        Range for the number of additional target points (min_range, max_range) if
        values are smaller than 1 these represent a percentage of points.

    is_all_trgts : bool, optional
        Whether to always use all the targets.
    """
    n_possible_points = X.size(1)
    intify = lambda x: ratio_to_int(x, n_possible_points)

    n_extra_trgts = random.randint(intify(range_extra_trgts[0]),
                                   intify(range_extra_trgts[1]))
    n_cntxts = random.randint(intify(range_cntxts[0]),
                              intify(range_cntxts[1]))
    n_trgts = (n_cntxts + n_extra_trgts) if is_all_trgts else n_possible_points

    # Sample locations of context and target points
    locations = np.random.choice(n_possible_points,
                                 size=n_trgts,
                                 replace=False)
    X_cntxt = X[:, locations[:n_cntxts], :]
    Y_cntxt = Y[:, locations[:n_cntxts], :]
    X_trgt = X[:, locations, :]
    Y_trgt = Y[:, locations, :]
    return X_cntxt, Y_cntxt, X_trgt, Y_trgt
