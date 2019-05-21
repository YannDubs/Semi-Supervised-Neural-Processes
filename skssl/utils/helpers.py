import sys
import math
import warnings

import torch.nn as nn


def ReversedConv2d(in_filter, out_filter, **kwargs):
    """Called the exact same way as Conv2d => with same in and out filter!"""
    return nn.ConvTranspose2d(out_filter, in_filter, **kwargs)


def ReversedLinear(in_size, out_size, **kwargs):
    """Called the exact same way as Linear => with same in and out dim!"""
    return nn.Linear(out_size, in_size, **kwargs)


def identity(x):
    """simple identity function"""
    return x


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
            i += count_layer(m)
    return i
