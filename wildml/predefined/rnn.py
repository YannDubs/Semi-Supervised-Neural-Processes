import torch
import torch.nn as nn

from wildml.utils.initialization import weights_init

__all__ = ["RNN"]


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size,
                 n_layers=1, dropout=0., bidirectional=False,
                 is_encoder=False, is_add_delta=False):
        super().__init__()
        self.is_encoder = is_encoder
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.is_add_delta = is_add_delta

        self.rnn = nn.GRU(input_size + is_add_delta, hidden_size, n_layers,
                          dropout=dropout, bidirectional=bidirectional,
                          batch_first=True)

        if not self.is_encoder:
            self.out = nn.Linear(hidden_size, output_size)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X, y=None):

        if self.is_add_delta:

            # assumes sorted !
            X_shift = X.clone()
            X_shift[..., 1:, :] = X[..., :-1, :]
            X = X - X_shift
            X = torch.cat([X, y], dim=-1)

        elif y is not None:
            # is there's y then that's the actual input and X is time
            X = y

        outputs, hidden = self.rnn(X)

        if self.is_encoder:
            if self.bidirectional:
                # sum bidirectional outputs
                outputs = (outputs[..., :self.hidden_size] +
                           outputs[..., self.hidden_size:])

            return outputs, hidden

        else:
            if self.bidirectional:
                # sum bidirectional outputs (only last of back and forth)
                outputs = (outputs[..., -1, :self.hidden_size] +
                           outputs[..., 0, self.hidden_size:])
            else:
                outputs = outputs[..., -1, :]

            outputs = self.out(outputs)
            return outputs
