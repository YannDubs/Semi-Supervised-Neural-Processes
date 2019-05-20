import sys
import math

import warnings


def is_valid_image_shape(shape):
    """Check if is a valid image shape."""
    if shape[1] != shape[2]:
        warnings.warn("Framework not tested for not squared images ... Shape = {}".format(shape))
    if shape[1] < 16:
        warnings.warn("Framework not tested for images with width < 16 ... Shape = {}".format(shape))
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
