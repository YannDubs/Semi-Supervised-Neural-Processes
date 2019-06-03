import sys
import math
import warnings

import numpy as np
import torch
import torch.nn as nn

from skorch.callbacks import Callback


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


def make_ssl_input(dataset, y=None):
    """
    Transform a dataset X to a variable that can be directly used like so:
    `NeuralNetEstimator.fit(*make_ssl_input(dataset))` for SSL. Namely, giving both
    `X` and `y` as input to `forward`. can also give a X and y.
    """
    if y is None:
        # forward takes y as argument and want to batch split it => put it in X
        # for X only get frst item of dataset => transformed data
        return ({'X': get_only_first_item(dataset),
                 "y": dataset.targets},
                dataset.targets)
    else:
        return ({'X': dataset, "y": y}, y)


def cont_tuple_to_tuple_cont(container):
    """COnverts a container (list, tuple, dict) of tuple to a tuple of container."""
    if isinstance(container, dict):
        return tuple(dict(zip(container, val)) for val in zip(*container.values()))
    elif isinstance(container, list) or isinstance(container, tuple):
        return tuple(zip(*container))
    else:
        raise ValueError("Unkown conatiner type: {}.".format(type(container)))


def split_labelled_unlabelled(to_split, y, is_ordered=False):
    """
    Split an array like, or a list / tuple / dictionary of arrays like in a
    labelled and unlabbeled part. If `is_ordered` then first n_lab are labelled
    and rest unlabbeled.
    """
    if isinstance(to_split, dict):
        lab_unlab = {k: split_labelled_unlabelled(v, y, is_ordered=is_ordered)
                     for k, v in to_split.items()}
    elif isinstance(to_split, list) or isinstance(to_split, tuple):
        lab_unlab = [split_labelled_unlabelled(i, y, is_ordered=is_ordered) for i in to_split]
    else:
        if is_ordered:
            n_lab = (y != -1).sum()
            return to_split[:n_lab], to_split[n_lab:]
        else:
            return to_split[y != -1], to_split[y == -1]

    lab, unlab = cont_tuple_to_tuple_cont(lab_unlab)
    return lab, unlab


def is_valid_image_shape(shape, min_width=0, max_width=float("inf")):
    """Check if is a valid image shape."""
    if shape[1] != shape[2]:
        warnings.warn("Framework not tested for not squared images ... Shape = {}".format(shape))
    if shape[1] > max_width:
        warnings.warn("Model only tested for images `width <= {}` ... Shape = {}".format(max_width, shape))
    if shape[1] < min_width:
        warnings.warn("Model only tested for images `width >= {}` ... Shape = {}".format(min_width, shape))
    if not is_power2(shape[1]):
        warnings.warn("Framework not tested for images with width not power of 2 ... Shape = {}".format(shape))


def closest_power2(n):
    """Return the closest power of 2 by checking whether the second binary number is a 1."""
    op = math.floor if bin(n)[3] != "1" else math.ceil
    return 2**(op(math.log(n, 2)))


def is_power2(num):
    """Check if is a power of 2."""
    return num != 0 and ((num & (num - 1)) == 0)


def check_import(module, to_use=None):
    """Check whether the given module is imported."""
    if module not in sys.modules:
        if to_use is None:
            error = '{} module not imported. Try "pip install {}".'.format(module, module)
            raise ImportError(error)
        else:
            error = 'You need {} to use {}. Try "pip install {}".'.format(module, to_use, module)
            raise ImportError(error)


def count_layer(module, layer):
    """Count number of times a layer is in a network."""
    i = 0
    if isinstance(module, layer):
        return 1
    else:
        for m in module.children():
            i += count_layer(m, layer)
    return i
