import torch
import torch.nn as nn

from skssl.predefined import MLP
from skssl.utils.initialization import weights_init

from .attention import get_attender

__all__ = ["SelfAttentionBlock", "SinusoidalEncodings"]


class SelfAttentionBlock(nn.Module):
    """Self Attention Layer.

    Parameters
    ----------
    x_dim : int
        Spatial input dimension.

    y_dim : int
        Value input dimension.

    out_dim : int
        Output dimension. If different than x_dim will do all the computation
        with a size of `x_dim` and add a linear layer at the end to reshape.

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
                 n_attn_layers=2,
                 attention="transformer",
                 is_normalize=True,
                 **kwargs):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.out_dim = out_dim

        self.is_reshape_y = self.y_dim != self.x_dim
        if self.is_reshape_y:
            self.reshape_y = MLP(self.y_dim, self.x_dim)

        self.attn_layers = nn.ModuleList([get_attender(attention, self.x_dim,
                                                       is_normalize=is_normalize,
                                                       **kwargs)
                                          for _ in range(n_attn_layers)])

        self.is_reshape_out = self.x_dim != self.out_dim
        if self.is_reshape_out:
            self.reshape_out = nn.Linear(x_dim, self.out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X, Y):
        if self.is_reshape_y:
            Y = self.reshape_y(Y)

        out = Y + X

        for attn_layer in self.attn_layers:
            out = attn_layer(out, out, out)

        if self.is_reshape_out:
            out = self.reshape_out(out)

        return out


class SinusoidalEncodings(nn.Module):
    """
    Converts a batch of N-dimensional spatial input X with values between `[-1,1]`
    to a batch of flat in vector splitted in N subvectors that encode the position via
    sinusoidal encodings.

    Parameters
    ----------
    x_dim : int
        Number of spatial inputs.

    out_dim : int
        size of output encoding. Each x_dim will have an encoding of size
        `out_dim//x_dim`.
    """

    def __init__(self, x_dim, out_dim):
        super().__init__()
        self.x_dim = x_dim
        # dimension of encoding for eacg x dimension
        self.sub_dim = out_dim // self.x_dim
        # in attention is all you need used 10000 but 512 dim, try to keep the
        # same ratio regardless of dim
        self._C = 10000 * (self.sub_dim / 512)**2

        if out_dim % x_dim != 0:
            raise ValueError("out_dim={} has to be dividable by x_dim={}.".format(out_dim, x_dim))
        if self.sub_dim % 2 != 0:
            raise ValueError("sum_dim=out_dim/x_dim={} has to be dividable by 2.".format(self.sub_dim))

        self._precompute_denom()

    def _precompute_denom(self):
        two_i_d = torch.arange(0, self.sub_dim, 2, dtype=torch.float) / self.sub_dim
        denom = torch.pow(self._C, two_i_d)
        denom = torch.repeat_interleave(denom, 2).unsqueeze(0)
        self.denom = denom.expand(1, self.x_dim, self.sub_dim)

    def forward(self, x):
        shape = x.shape
        # flatten besides last dim
        x = x.view(-1, shape[-1])
        # will only be passed once to GPU because precomputed
        self.denom = self.denom.to(x.device)
        # put x in a range which is similar to positions in NLP [1,201]
        #x = (x.unsqueeze(-1) + 1)*100 + 1
        x = x.unsqueeze(-1)
        out = x / self.denom
        out[:, :, 0::2] = torch.sin(out[:, :, 0::2])
        out[:, :, 1::2] = torch.cos(out[:, :, 1::2])
        # concatenate all different sinusoidal encodings for each x_dim
        # and unflatten
        out = out.view(*shape[:-1], self.sub_dim * self.x_dim)
        return out
