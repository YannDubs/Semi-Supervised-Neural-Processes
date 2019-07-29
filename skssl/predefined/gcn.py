import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import GCNConv, GATConv

from skssl.utils.initialization import weights_init

__all__ = ["GCN", "UnetGCN", "GAT"]


class GCN(torch.nn.Module):
    def __init__(self, n_channels,
                 Conv=GCNConv,
                 n_layers=3,
                 is_res=True,
                 Activation=nn.ReLU,
                 Normalization=nn.Identity,
                 _is_summary=False,
                 **kwargs):
        super().__init__()

        self.n_channels = n_channels,
        self._is_summary = _is_summary
        self.n_layers = n_layers
        self.activation_ = Activation(inplace=True)
        self.is_res = is_res
        self.in_out_channels = self._get_in_out_channels(n_channels, n_layers)

        self.convs = nn.ModuleList([Conv(in_chan, out_chan, **kwargs)
                                    for in_chan, out_chan in self.in_out_channels])

        self.norms = nn.ModuleList([Normalization(out_chan)
                                    for _, out_chan in self.in_out_channels])

    def reset_parameters(self):
        weights_init(self)
        for conv in self.convs:
            conv.reset_parameters()

    def _get_in_out_channels(self, n_channels, n_layers):
        """Return a list of tuple of input and output channels."""

        if isinstance(n_channels, int):
            channel_list = [n_channels] * n_layers
        else:
            channel_list = list(n_channels)
            assert len(channel_list) == n_layers + 1

        return list(zip(channel_list, channel_list[1:]))

    def forward(self, X):

        X = self.apply_convs(X)
        X, summary = X

        if not self._is_summary:
            summary = None

        return X, summary

    def apply_convs(self, data):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if self.is_res:
                # normalization and residual
                data.x = self.activation_(norm(conv(data.x, data.edge_index))) + data.x
            else:
                data.x = self.activation_(norm(conv(data.x, data.edge_index)))

        return data, None


class UnetGCN(GCN):
    def __init__(self, n_channels,
                 Conv=GCNConv,
                 Pool=torch_geometric.nn.TopKPooling,
                 is_double_conv=False,
                 max_nchannels=512,
                 bottleneck=None,
                 n_layers=5,
                 is_force_same_bottleneck=False,
                 **kwargs):

        self.is_double_conv = is_double_conv
        self.max_nchannels = max_nchannels
        super().__init__(n_channels, Conv,
                         n_layers=n_layers,
                         is_res=False,
                         **kwargs)

        pool_in_chan = [max(self.n_channels, self.max_nchannels)
                        for i in range(self.n_layers // 2 + 1)]
        self.pools = nn.ModuleList([Pool(in_chan) for in_chan in pool_in_chan])
        self.is_force_same_bottleneck = is_force_same_bottleneck

    def apply_convs(self, X):
        x, edge_index, batch = X.x, X.edge_index, X.batch

        n_blocks = self.n_layers // 2 if self.is_double_conv else self.n_layers
        n_down_blocks = n_blocks // 2
        residuals = [None] * n_down_blocks
        edges = [None] * n_down_blocks
        perms = [None] * n_down_blocks

        # Down
        for i in range(n_down_blocks):
            x = self._apply_conv_block_i(x, edge_index, i)
            residuals[i] = x
            x, edge_index, _, batch, perm, _ = self.pools(x, edge_index, batch=batch)
            #edges[i] = edge_index
            perms[i] = perm

        # Bottleneck
        x = self._apply_conv_block_i(x, edge_index, i)
        summary = torch_geometric.nn.global_mean_pool(x, batch)  # summary before forcing same bottleneck!

        if self.is_force_same_bottleneck and self.training:
            # if all the batches are from the same function then use the same
            # botlleneck for all to be sure that use the summary
            # in this case the batch should bea concatenated same batch
            # only force during training
            batch_size = X.size(0)
            batch_1 = X[:batch_size // 2, ...]
            batch_2 = X[batch_size // 2:, ...]
            X_mean = (batch_1 + batch_2) / 2
            X = torch.cat([X_mean, X_mean], dim=0)

        # Up
        for i in range(n_down_blocks + 1, n_blocks):
            up = torch.zeros_like(x)
            up[perms[n_down_blocks - i]] = residuals[n_down_blocks - i]
            x = torch.cat((x, up), dim=1)  # conncat on channels
            x = self._apply_conv_block_i(x, edge_index, i)

        X.x, X.edge_index, X.batch

        return X, summary

    def _apply_conv_block_i(self, x, edge_index, i):
        """Apply the i^th convolution block."""
        if self.is_double_conv:
            i *= 2

        x = self.activation_(self.norms[i](self.convs[i](x, edge_index)))

        if self.is_double_conv:
            x = self.activation_(self.norms[i + 1](self.convs[i + 1](x, edge_index)))

        return x

    def _get_in_out_channels(self, n_channels, n_layers):
        """Return a list of tuple of input and output channels for a Unet."""
        factor_chan = 1 if self.bottleneck == "channels" else 2

        if self.is_double_conv:
            assert n_layers % 2 == 0, "n_layers={} not even".format(n_layers)
            # e.g. if n_channels=16, n_layers=10: [16, 32, 64]
            channel_list = [factor_chan**i * n_channels for i in range(n_layers // 4 + 1)]
            # e.g.: [16, 16, 32, 32, 64, 64]
            channel_list = [i for i in channel_list for _ in (0, 1)]
            # e.g.: [16, 16, 32, 32, 64, 64, 64, 32, 32, 16, 16]
            channel_list = channel_list + channel_list[-2::-1]
            # bound max number of channels by self.max_nchannels
            channel_list = [min(c, self.max_nchannels) for c in channel_list]
            # e.g.: [..., (32, 32), (32, 64), (64, 64), (64, 32), (32, 32), (32, 16) ...]
            in_out_channels = super()._get_in_out_channels(channel_list, n_layers)
            # e.g.: [..., (32, 32), (32, 64), (64, 64), (128, 32), (32, 32), (64, 16) ...]
            # due to concat
            idcs = slice(len(in_out_channels) // 2 + 1, len(in_out_channels), 2)
            in_out_channels[idcs] = [(in_chan * 2, out_chan)
                                     for in_chan, out_chan in in_out_channels[idcs]]
        else:
            assert n_layers % 2 == 1, "n_layers={} not odd".format(n_layers)
            # e.g. if n_channels=16, n_layers=5: [16, 32, 64]
            channel_list = [factor_chan**i * n_channels for i in range(n_layers // 2 + 1)]
            # e.g.: [16, 32, 64, 64, 32, 16]
            channel_list = channel_list + channel_list[::-1]
            # bound max number of channels by self.max_nchannels
            channel_list = [min(c, self.max_nchannels) for c in channel_list]
            # e.g.: [(16, 32), (32,64), (64, 64), (64, 32), (32, 16)]
            in_out_channels = super()._get_in_out_channels(channel_list, n_layers)
            # e.g.: [(16, 32), (32,64), (64, 64), (128, 32), (64, 16)] due to concat
            idcs = slice(len(in_out_channels) // 2 + 1, len(in_out_channels))
            in_out_channels[idcs] = [(in_chan * 2, out_chan)
                                     for in_chan, out_chan in in_out_channels[idcs]]

        return in_out_channels


class GAT(nn.Module):
    def __init__(self, in_channels, out_channels=None, dim=32, heads=8, dropout=0.3, n_layers=2):
        super().__init__()
        self.dropout = dropout
        if out_channels is None:
            out_channels = in_channels

        kwargs = dict(heads=heads, dropout=dropout, concat=True)

        self.convs = nn.ModuleList([GATConv(in_channels, dim, **kwargs)
                                    ] + [GATConv(dim * heads, dim, **kwargs)
                                         for _ in range(n_layers - 2)])

        self.conv_out = GATConv(
            dim * heads, out_channels, heads=heads, concat=False, dropout=dropout)

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index))

        x = self.conv_out(x, data.edge_index)
        data.x = x

        return data, None
