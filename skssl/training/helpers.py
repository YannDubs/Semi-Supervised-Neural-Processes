import numpy as np
import torch

from skorch.callbacks import Callback

from skssl.utils.helpers import cont_tuple_to_tuple_cont


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
    if isinstance(dataset, dict):
        y = dataset["y"]
        dataset = dataset["X"]

    if y is None:
        # forward takes y as argument and want to batch split it => put it in X
        # for X only get frst item of dataset => transformed data
        return ({'X': get_only_first_item(dataset),
                 "y": dataset.targets},
                dataset.targets)
    else:
        return ({'X': dataset, "y": y}, y)


def split_labelled_unlabelled(to_split, y, is_stacked=False):
    """
    Split an array like, or a list / tuple / dictionary of arrays like in a
    labelled and unlabbeled part. If `is_stacked` then first array on dim=1 is
    labeleld and the rest unlabelled.
    """
    if isinstance(to_split, dict):
        lab_unlab = {k: split_labelled_unlabelled(v, y, is_stacked=is_stacked)
                     for k, v in to_split.items()}
    elif isinstance(to_split, list) or isinstance(to_split, tuple):
        lab_unlab = [split_labelled_unlabelled(i, y, is_stacked=is_stacked) for i in to_split]
    else:
        if is_stacked:
            return to_split[:, 0, ...], to_split[:, 1:, ...]
        else:
            return to_split[y != -1], to_split[y == -1]

    lab, unlab = cont_tuple_to_tuple_cont(lab_unlab)
    return lab, unlab
