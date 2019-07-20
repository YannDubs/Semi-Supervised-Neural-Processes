import numpy as np
import sys
import math
import warnings

from functools import reduce
import operator


def indep_shuffle_(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.

    Credits : https://github.com/numpy/numpy/issues/5173
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])


def ratio_to_int(percentage, max_val):
    """Converts a ratio to an integer if it is smaller than 1."""
    if 1 <= percentage <= max_val:
        out = percentage
    elif 0 <= percentage < 1:
        out = percentage * max_val
    else:
        raise ValueError("percentage={} outside of [0,{}].".format(percentage, max_val))

    return int(out)


def prod(iterable):
    """COmpute the product of all elements in an iterable."""
    return reduce(operator.mul, iterable, 1)


def mean(iterable):
    """COmpute the mean of all elements in an iterable."""
    return sum(iterable) / len(iterable)


def cont_tuple_to_tuple_cont(container):
    """COnverts a container (list, tuple, dict) of tuple to a tuple of container."""
    if isinstance(container, dict):
        return tuple(dict(zip(container, val)) for val in zip(*container.values()))
    elif isinstance(container, list) or isinstance(container, tuple):
        return tuple(zip(*container))
    else:
        raise ValueError("Unkown conatiner type: {}.".format(type(container)))


def tuple_cont_to_cont_tuple(tuples):
    """Converts a tuple of containers (list, tuple, dict) to a container of tuples."""
    if isinstance(tuples[0], dict):
        # assumes keys are correct
        return {k: tuple(dic[k] for dic in tuples) for k in tuples[0].keys()}
    elif isinstance(tuples[0], list):
        return list(zip(*tuples))
    elif isinstance(tuples[0], tuple):
        return tuple(zip(*tuples))
    else:
        raise ValueError("Unkown conatiner type: {}.".format(type(tuples[0])))


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


class HyperparameterInterpolator:
    """Helper class to compute the value of a hyperparameter at each training step.

    Parameters
    ----------
    initial_value: float
        Initial value of the hyperparameter.

    final_value: float
        Final value of the hyperparameter.

    N_steps_interpolate: int
        Number of training steps before reaching the `final_value`.

    Start_step: int, optional
        Number of steps to wait for before starting annealing. During the waiting time,
        the hyperparameter will be `default`.

    Default: float, optional
        Default hyperparameter value that will be used for the first `start_step`s. If
        `None` uses `initial_value`.

    mode: {"linear", "geometric"}, optional
        Interpolation mode.
    """

    def __init__(self, initial_value, final_value, n_steps_interpolate,
                 start_step=0,
                 default=None,
                 mode="linear"):

        self.initial_value = initial_value
        self.final_value = final_value
        self.n_steps_interpolate = n_steps_interpolate
        self.start_step = start_step
        self.default = default if default is not None else self.initial_value
        self.mode = mode.lower()

        if self.mode == "linear":
            delta = (self.final_value - self.initial_value)
            self.factor = delta / self.n_steps_interpolate
        elif self.mode == "geometric":
            delta = (self.final_value / self.initial_value)
            self.factor = delta ** (1 / self.n_steps_interpolate)
        else:
            raise ValueError("Unkown mode : {}.".format(mode))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the interpolator."""
        self.n_training_calls = 0

    @property
    def is_annealing(self):
        return (self.start_step <= self.n_training_calls) and (
            self.n_training_calls <= (self.n_steps_interpolate + self.start_step))

    def __call__(self, is_update):
        """Return the current value of the hyperparameter.

        Parameters
        ----------
        Is_update: bool
            Whether to update the hyperparameter.
        """
        if is_update:
            self.n_training_calls += 1

        if self.start_step >= self.n_training_calls:
            return self.default

        n_actual_training_calls = self.n_training_calls - self.start_step

        if self.is_annealing:
            current = self.initial_value
            if self.mode == "geometric":
                current *= (self.factor ** n_actual_training_calls)
            elif self.mode == "linear":
                current += self.factor * n_actual_training_calls
        else:
            current = self.final_value

        return current


def rescale_range(X, old_range, new_range):
    """Rescale X linearly to be in `new_range` rather than `old_range`."""
    old_min = old_range[0]
    new_min = new_range[0]
    old_delta = old_range[1] - old_min
    new_delta = new_range[1] - new_min
    return (((X - old_min) * new_delta) / old_delta) + new_min
