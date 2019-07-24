import torch
import torch.nn as nn

from skssl.utils.initialization import weights_init


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size,
                 n_layers=1, dropout=0.5, bidirectional=True,
                 is_return_hidden=False, is_add_delta=True):
        super().__init__()
        self.is_return_hidden = is_return_hidden
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.is_add_delta = is_add_delta

        self.rnn = nn.GRU(input_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=bidirectional, batch_first=True)

        if output_size is not None:
            self.out = nn.Linear(hidden_size, output_size)
        else:
            self.out = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X, y=None):
        if self.is_add_delta:
            # assumes sorted !
            X_shift = X.clone()
            X_shift[..., 1:] = X[..., :-1]
            X = X - X_shift
            X = torch.cat([X, y], dim=-1)

        elif y is not None:
            # is there's y then that's the actual input and X is time
            X = y

        outputs, hidden = self.gru(X)

        if self.bidirectional:
            # sum bidirectional outputs
            outputs = (outputs[..., :self.hidden_size] +
                       outputs[..., self.hidden_size:])

        outputs = self.out(outputs)

        if self.is_return_hidden:
            return outputs, hidden

        return outputs
