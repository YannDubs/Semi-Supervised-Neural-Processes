import warnings

import torch
import torch.nn as nn

from essl.utils.initialization import linear_init


__all__ = ["MLP"]


class MLP(nn.Module):
    """General MLP class.

    Parameters
    ----------
    input_size: int

    output_size: int

    hidden_size: int or list, optional
        Number of hidden neurones. If list, `n_hidden_layers` will be `len(n_hidden_layers)`.

    n_hidden_layers: int, optional
        Number of hidden layers.

    activation: callable, optional
        Activation function. E.g. `nn.RelU()`.

    is_bias: bool, optional
        Whether to use biaises in the hidden layers.

    dropout: float, optional
        Dropout rate.

    is_force_hid_larger : bool, optional
        Whether to force the hidden dimension to be larger or equal than in or out.

    is_res : bool, optional
        Whether to use residual connections.
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=32,
        n_hidden_layers=1,
        activation=nn.ReLU(),
        is_bias=True,
        dropout=0,
        is_force_hid_larger=False,
        is_res=False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.is_res = is_res

        if self.n_hidden_layers == 0:
            self.to_hidden = nn.Linear(self.input_size, self.output_size, bias=is_bias)
            self.out = nn.Identity()
            return

        if isinstance(self.hidden_size, int):
            if is_force_hid_larger and self.hidden_size < min(self.output_size, self.input_size):
                self.hidden_size = min(self.output_size, self.input_size)
                txt = "hidden_size={} smaller than output={} and input={}. Setting it to {}."
                warnings.warn(txt.format(hidden_size, output_size, input_size, self.hidden_size))

            self.hidden_size = [self.hidden_size] * self.n_hidden_layers
        else:
            self.n_hidden_layers = len(self.hidden_size)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.activation = activation

        self.to_hidden = nn.Linear(self.input_size, self.hidden_size[0], bias=is_bias)
        self.linears = nn.ModuleList(
            [
                nn.Linear(in_size, out_size, bias=is_bias)
                for in_size, out_size in zip(self.hidden_size[:][:-1], self.hidden_size[1:])
                # dirty [:] because omegaconf does not accept [:-1] directly
            ]
        )
        self.out = nn.Linear(self.hidden_size[-1], self.output_size, bias=is_bias)

        self.reset_parameters()

    def forward(self, x):
        out = self.to_hidden(x)

        if self.n_hidden_layers == 0:
            return out

        out = self.activation(out)
        x = self.dropout(out)

        for linear in self.linears:
            out = linear(x)
            out = self.activation(out)
            if self.is_res:
                out = out + x
            out = self.dropout(out)
            x = out

        out = self.out(x)
        return out

    def reset_parameters(self):
        linear_init(self.to_hidden, activation=self.activation)
        for lin in self.linears:
            linear_init(lin, activation=self.activation)
        linear_init(self.out)
