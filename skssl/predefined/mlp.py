import torch.nn as nn

from skssl.utils.initialization import linear_init
from skssl.utils.helpers import identity


class MLP(nn.Module):
    """General MLP class.

    Parameters
    ----------
    input_size : int

    output_size : int

    hidden_size (int, optional):
        Number of hidden neurones. Forces it to be between [input_size, output_size].

    activation : torch.nn.modules.activation, optional
        Unitialized activation class.
    """

    def __init__(self, input_size, output_size,
                 hidden_size=32,
                 activation=nn.ReLU,
                 bias=True,
                 dropout=0):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = min(self.input_size, max(hidden_size, self.output_size))

        self.to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        self.dropout = (nn.Dropout(p=dropout) if dropout > 0 else identity)
        self.activation = activation()  # cannot be a function from Functional but class
        self.out = nn.Linear(self.hidden_size, self.output_size, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        y = self.to_hidden(x)
        y = self.dropout(y)
        y = self.activation(y)
        y = self.out(y)
        return y

    def reset_parameters(self):
        linear_init(self.to_hidden, activation=self.activation)
        linear_init(self.out)
