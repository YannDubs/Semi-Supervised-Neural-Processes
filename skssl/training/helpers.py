import random

import numpy as np
import torch

from skorch.callbacks import Callback

from skssl.utils.helpers import cont_tuple_to_tuple_cont, ratio_to_int


class FixRandomSeed(Callback):
    """
    Callback to have a deterministic behavior.
    Credits: https://github.com/skorch-dev/skorch/issues/280
    """

    def __init__(self, seed=123, is_cudnn_deterministic=False, verbose=0):
        self.seed = seed
        self.is_cudnn_deterministic = is_cudnn_deterministic
        self.verbose = verbose

    def initialize(self):
        if self.verbose > 0:
            print("setting random seed to: ", self.seed)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        try:
            random.seed(self.seed)
        except NameError:
            import random
            random.seed(self.seed)

        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = self.is_cudnn_deterministic


def get_only_first_item(to_index):
    """Helper function to make a class `to_index` return `to_index[i][0]` when indexed."""
    class FirstIndex:
        def __init__(self, to_index):
            self.to_index = to_index

        def __getitem__(self, i):
            return self.to_index[i][0]

        def __len__(self):
            return len(self.to_index)

    return FirstIndex(to_index)


def make_Xy_input(dataset, y=None):
    """
    Transform a dataset X to a variable that can be directly used like so:
    `NeuralNetEstimator.fit(*make_Xy_input(dataset))` when both `X` and `y`
    should be inputs to `forward`. Can also give a X and y.
    """
    if isinstance(dataset, dict):
        y = dataset["y"]
        X = dataset["X"]
    elif isinstance(dataset, torch.utils.data.Dataset):
        if y is None:
            try:
                y = dataset.targets
            except AttributeError:
                y = dataset.y  # skorch datasets
        X = get_only_first_item(dataset)
    else:
        # array-like or tensor
        X = dataset

    return ({'X': X, "y": y}, y)


def split_labelled_unlabelled(to_split, y):
    """
    Split an array like, or a list / tuple / dictionary of arrays like in a
    labelled and unlabbeled part.
    """
    if isinstance(to_split, dict):
        lab_unlab = {k: split_labelled_unlabelled(v, y)
                     for k, v in to_split.items()}
        if len(lab_unlab) == 0:
            return {}, {}
    elif isinstance(to_split, list) or isinstance(to_split, tuple):
        lab_unlab = [split_labelled_unlabelled(i, y) for i in to_split]
        if len(lab_unlab) == 0:
            return [], []
    else:
        return to_split[y != -1], to_split[y == -1]

    lab, unlab = cont_tuple_to_tuple_cont(lab_unlab)
    return lab, unlab


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
    X : torch.Tensor
        Shape (batch_size, num_points, x_dim)

    Y : torch.Tensor
        Shape (batch_size, num_points, y_dim)

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
