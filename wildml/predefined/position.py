import torch
import torch.nn as nn

__all__ = ["RelativeSinusoidalEncodings", "SinusoidalEncodings"]


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
        # in "attention is all you need" used 10000 but 512 dim, try to keep the
        # same ratio regardless of dim
        self._C = 10000 * (self.sub_dim / 512) ** 2

        if out_dim % x_dim != 0:
            raise ValueError("out_dim={} has to be dividable by x_dim={}.".format(out_dim, x_dim))
        if self.sub_dim % 2 != 0:
            raise ValueError(
                "sum_dim=out_dim/x_dim={} has to be dividable by 2.".format(self.sub_dim)
            )

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
        # put x in a range which is similar to positions in NLP [1,51]
        x = (x.unsqueeze(-1) + 1) * 25 + 1
        out = x / self.denom
        out[:, :, 0::2] = torch.sin(out[:, :, 0::2])
        out[:, :, 1::2] = torch.cos(out[:, :, 1::2])
        # concatenate all different sinusoidal encodings for each x_dim
        # and unflatten
        out = out.view(*shape[:-1], self.sub_dim * self.x_dim)
        return out


class RelativeSinusoidalEncodings(nn.Module):
    """Return relative positions of inputs between [-1,1]."""

    def __init__(self, x_dim, out_dim, window_size=2):
        super().__init__()
        self.pos_encoder = SinusoidalEncodings(x_dim, out_dim)
        self.weight = nn.Linear(out_dim, out_dim, bias=False)
        self.window_size = window_size
        self.out_dim = out_dim

    def forward(self, keys_pos, queries_pos):
        # size=[batch_size, n_queries, n_keys, x_dim]
        diff = (keys_pos.unsqueeze(1) - queries_pos.unsqueeze(2)).abs()

        # the abs differences will be between between 0, self.window_size
        # we multipl by 2/self.window_size then remove 1 to  be [-1,1] which is
        # the range for `SinusoidalEncodings`
        scaled_diff = diff * 2 / self.window_size - 1
        out = self.weight(self.pos_encoder(scaled_diff))

        # set to 0 points that are further than window for extap
        out = out * (diff < self.window_size).float()

        return out

