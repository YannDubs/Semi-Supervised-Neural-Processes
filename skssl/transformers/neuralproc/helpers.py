import math

import torch
import torch.nn as nn

from skssl.predefined import MLP
from skssl.utils.initialization import weights_init

from .attention import get_attender

__all__ = ["SelfAttentionEncoder", "SinusoidalEncodings"]


class SelfAttentionEncoder(nn.Module):
    """Self Encoder Layer.

    Parameters
    ----------
    x_dim : int
        Spatial input dimension.

    y_dim : int
        Value input dimension.

    out_dim : int
        Output dimension. Currently only accepts if same value as x_dim.

    PreEncoder : nn.Module, optional
        Transformation of the inputs before the self attention.

    n_attn_layers : int, optional
        Number of self attention layers.

    attention : {'multiplicative', "additive", "scaledot", "multihead", "manhattan",
                "euclidean", "cosine", "transformer"}, optional
        Type of attention to use. More details in `get_attender`.

    is_normalize : bool, optional
        Whether qttention weights should sum to 1 (using softmax). If not weights
        will be in [0,1] but not necessarily sum to 1.
    """

    def __init__(self, x_dim, y_dim, out_dim,
                 YEncoder=MLP,
                 n_attn_layers=2,
                 attention="scaledot",
                 is_normalize=True):
        super().__init__()
        self.x_dim = x_dim
        self.out_dim = out_dim
        self.y_dim = y_dim
        self.y_encoder = YEncoder(self.y_dim, self.out_dim)
        self.attn_layers = nn.ModuleList([get_attender(attention, self.out_dim,
                                                       is_normalize=is_normalize)
                                          for _ in range(n_attn_layers)])

        self.reset_parameters()

        if x_dim != out_dim:
            raise ValueError("Currently only accepts x_dim=out_dim but {}!={}.".format(x_dim, out_dim))

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X, Y):
        Y = self.y_encoder(Y)
        out = Y + X

        for attn_layer in self.attn_layers:
            out = attn_layer(out, out, out)

        return out


class SinusoidalEncodings(nn.Module):
    """
    Converts spatial N-dimensional spatial input with values between `[-1,1]`
    to a flat in vector splitted in N subvectors that encode the position via
    sinusoidal encodings.

    """

    def __init__(self, x_dim, out_dim):
        self.x_dim = x_dim
        self.out_dim = out_dim
        self._precompute_denom()

        if out_dim % x_dim != 0:
            raise ValueError("out_dim={} has to be dividable by x_dim={}.".format(out_dim, x_dim))

    def _precompute_denom(self):
        two_i_d = torch.arange(0, self.out_dim, 2, dtype=torch.float) / self.out_dim
        # in log domain for math stability
        denom = torch.exp(two_i_d * math.log(100))
        denom = torch.repeat_interleave(denom, 2)
        self.denom = denom.view(self.x_dim, self.out_dim // self.x_dim)

    def forward(self, x):
        batch_size = x.size()
        # will only be passed once to GPU because precomputed
        self.denom = self.denom.to(x.device)
        # add 1 to not have negative values as positions (input is in [-1,1])
        x = x.unsqueeze(-1) + 1
        out = self.denom.clone()
        out[:, 0::2] = torch.sin(x / self.denom[:, 0::2])
        out[:, 1::2] = torch.cos(x / self.denom[:, 1::2])
        return out.view(self.out_dim)
