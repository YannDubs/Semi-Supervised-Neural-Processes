import torch
import torch.nn as nn

from skssl.utils.initialization import weights_init
from .mlp import MLP


class MergeFlatAndNotFlatInputs(nn.Module):
    """
    Extend a module which takes takes a non flat input (e.g. images) such that it can
    also take a flat input. The output of the given module will be concatenated with
    the flat input and passed to a MLP.

    Parameters
    ----------
    NonFlatModule: nn.Module
        Module which takes in non flat inputs (e.g. images).

    x_shape: array-like
        Shape of a single non flat example.

    flat_dim: int
        Dimensionality of the flat inputs.

    n_out: int
        Size of ouput.

    hidden_size: int, optional
        Hidden size of the MLP which puts together the output of NonFlatModule and the
        flat input. If -1 uses n_out.

    n_hidden_layers: int, optional
        Number of hidden layers of the MLP which puts together the output of NonFlatModule
        and the flat input.

    kwargs:
        Additional arguments to NonFlatModule.
    """

    def __init__(self, NonFlatModule, x_shape, flat_dim, n_out,
                 hidden_size=-1,
                 n_hidden_layers=1,
                 **kwargs):
        super().__init__()
        hidden_size = hidden_size if hidden_size != -1 else n_out
        self.non_flat_module = NonFlatModule(x_shape, hidden_size, **kwargs)
        self.mixer = MLP(hidden_size + flat_dim, n_out,
                         hidden_size=hidden_size,
                         n_hidden_layers=n_hidden_layers)
        self.reset_parameters()

    def forward(self, x, flat_input):
        non_flat_out = self.non_flat_module(x)
        out = self.mixer(torch.cat((non_flat_out, flat_input), dim=1))
        return out

    def reset_parameters(self):
        weights_init(self)


class MergeFlatInputs(nn.Module):
    """
    Extend a module to take 2 flat inputs. It simply returns
    the concatenated flat inputs to the module `module({x1; x2})`.

    Parameters
    ----------
    FlatModule: nn.Module
        Module which takes a non flat inputs.

    x1_dim: int
        Dimensionality of the first flat inputs.

    x2_dim: int
        Dimensionality of the second flat inputs.

    n_out: int
        Size of ouput.

    kwargs:
        Additional arguments to FlatModule.
    """

    def __init__(self, FlatModule, x1_dim, x2_dim, n_out, **kwargs):
        super().__init__()
        self.flat_module = FlatModule(x1_dim + x2_dim, n_out, **kwargs)
        self.reset_parameters()

    def forward(self, x1, x2):
        return self.flat_module(torch.cat((x1, x2), dim=1))

    def reset_parameters(self):
        weights_init(self)


def add_flat_input(module, **kwargs):
    """
    Extend a module to accept an additional flat input. I.e. the output should
    be called by `add_flat_input(module)(x_shape, flat_dim, n_out, **kwargs)`.

    Notes
    -----
    - if x_shape is an integer (i.e. x is already flat), it simply returns
    the concatenated flat inputs to the module `module({x; flat_input})`.
    - if x_shape is not an integer (i.e. non flat input), it concatenates the
    output of `module(x)` with the flat input and passes it through a MLP.
    """
    def added_flat_input(x_shape, flat_dim, n_out, **kwargs2):
        if isinstance(x_shape, int):
            return MergeFlatInputs(module, x_shape, flat_dim, n_out, **kwargs2, **kwargs)
        else:
            return MergeFlatAndNotFlatInputs(module, x_shape, flat_dim, n_out, **kwargs2, **kwargs)
    return added_flat_input
