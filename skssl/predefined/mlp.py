import warnings

import torch.nn as nn

from skssl.utils.initialization import linear_init
from skssl.utils.torchextend import identity

__all__ = ["MLP","DeepMLP"]


def DeepMLP(*args):
    return MLP(*args, hidden_size=128, n_hidden_layers=3)


class MLP(nn.Module):
    """General MLP class.

    Parameters
    ----------
    input_size: int

    output_size: int

    hidden_size: int, optional
        Number of hidden neurones.

    n_hidden_layers: int, optional
        Number of hidden layers.

    activation: torch.nn.modules.activation, optional
        Unitialized activation class.

    bias: bool, optional
        Whether to use biaises in the hidden layers.

    dropout: float, optional
        Dropout rate.
    """

    def __init__(self, input_size, output_size,
                 hidden_size=32,
                 n_hidden_layers=1,
                 activation=nn.ReLU,
                 bias=True,
                 dropout=0):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers

        if self.hidden_size < min(self.output_size, self.input_size):
            self.hidden_size = min(self.output_size, self.input_size)
            txt = "hidden_size={} smaller than output={} and input={}. Setting it to {}."
            warnings.warn(txt.format(hidden_size, output_size, input_size, self.hidden_size))

        self.dropout = (nn.Dropout(p=dropout) if dropout > 0 else identity)
        self.activation = activation()  # cannot be a function from Functional but class

        self.to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        self.linears = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
                                      for _ in range(self.n_hidden_layers - 1)])
        self.out = nn.Linear(self.hidden_size, self.output_size, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        out = self.to_hidden(x)
        out = self.activation(out)
        out = self.dropout(out)

        for linear in self.linears:
            out = linear(out)
            out = self.activation(out)
            out = self.dropout(out)

        out = self.out(out)
        return out

    def reset_parameters(self):
        linear_init(self.to_hidden, activation=self.activation)
        for lin in self.linears:
            linear_init(lin, activation=self.activation)
        linear_init(self.out)
