from functools import partial

import torch.nn as nn

from .cnn import get_Cnn
from .mlp import MLP
from .meta import merge_flat_input, discard_ith_arg
from .attention import get_Attender
from .selfattn import SelfAttention

__all__ = ["get_predefined", "try_get_predefined"]


def try_get_predefined(d, **kwargs):
    """Tries to get a predefined module, given a dicttionary of all arguments, if not returns it."""
    try:
        return get_predefined(**d, **kwargs)
    except TypeError:
        return d


# TO DOC
def get_predefined(name, meta_kwargs={}, **kwargs):
    """Helper function which returns unitialized common neural networks."""
    name = name.lower()

    if name == "cnn":
        Module = get_Cnn(**kwargs)
    elif name == "mlp":
        Module = partial(MLP, **kwargs)
    elif name == "attention":
        Module = get_Attender(**kwargs)
    elif name == "selfattention":
        Module = partial(SelfAttention, **kwargs)
    elif name == "identity":
        Module = nn.Identity
    elif name == "linear":
        Module = partial(nn.Linear, **kwargs)
    elif name is None:
        return None
    elif not isinstance(name, str):
        Module = name
    else:
        raise ValueError(name)

    if len(meta_kwargs) > 0:
        meta_kwargs = meta_kwargs.copy()
        mode = meta_kwargs.pop("mode")
        if mode == "merge":
            Module = merge_flat_input(Module, **meta_kwargs)
        elif mode == "discard":
            Module = discard_ith_arg(Module, **meta_kwargs)
        else:
            raise ValueError("Unkown meta type {}.".format(mode))

    return Module

