import warnings

import torch.nn as nn

from skssl.utils.initialization import linear_init


__all__ = ["MLP"]


class MLP(nn.Module):
    """General MLP class with residual.

    Parameters
    ----------
    input_size: int

    output_size: int

    hidden_size: int, optional
        Number of hidden neurones.

    n_hidden_layers: int, optional
        Number of hidden layers.

    Activation: torch.nn.modules.activation, optional
        Unitialized activation class.

    bias: bool, optional
        Whether to use biaises in the hidden layers.

    dropout: float, optional
        Dropout rate.

    is_force_hid_smaller : bool, optional
        Whether to force the hidden dimensions to be smaller than in and out.

    is_res : bool, optional
        Whether to use residual connections.
    """

    def __init__(self, input_size, output_size,
                 hidden_size=32,
                 n_hidden_layers=1,
                 Activation=nn.ReLU,
                 bias=True,
                 dropout=0,
                 is_force_hid_smaller=False,
                 is_res=False):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.is_res = is_res

        if is_force_hid_smaller and self.hidden_size > max(self.output_size, self.input_size):
            self.hidden_size = max(self.output_size, self.input_size)
            txt = "hidden_size={} larger than output={} and input={}. Setting it to {}."
            warnings.warn(txt.format(hidden_size, output_size, input_size, self.hidden_size))
        elif self.hidden_size < min(self.output_size, self.input_size):
            self.hidden_size = min(self.output_size, self.input_size)
            txt = "hidden_size={} smaller than output={} and input={}. Setting it to {}."
            warnings.warn(txt.format(hidden_size, output_size, input_size, self.hidden_size))

        self.dropout = (nn.Dropout(p=dropout) if dropout > 0 else nn.Identity())
        self.activation_ = Activation(inplace=True)

        self.to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        self.linears = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
                                      for _ in range(self.n_hidden_layers - 1)])
        self.out = nn.Linear(self.hidden_size, self.output_size, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        out = self.to_hidden(x)
        self.activation_(out)
        x = self.dropout(out)

        for linear in self.linears:
            out = linear(x)
            self.activation_(out)
            if self.is_res:
                out = out + x
            out = self.dropout(out)
            x = out

        out = self.out(x)
        return out

    def reset_parameters(self):
        linear_init(self.to_hidden, activation=self.activation_)
        for lin in self.linears:
            linear_init(lin, activation=self.activation_)
        linear_init(self.out)
