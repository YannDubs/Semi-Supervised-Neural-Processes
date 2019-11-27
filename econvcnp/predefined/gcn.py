import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_sparse import spspmm
from torch_geometric.nn import GCNConv, GATConv, TopKPooling
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils.repeat import repeat
from torch_geometric.utils.num_nodes import maybe_num_nodes

from econvcnp.utils.initialization import weights_init

__all__ = ["GCN", "UnetGCN", "GAT", "GraphUNet"]


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

        self.n_channels = n_channels
        self._is_summary = _is_summary
        self.n_layers = n_layers
        self.activation_ = Activation(inplace=True)
        self.is_res = is_res
        self.in_out_channels = self._get_in_out_channels(n_channels, n_layers)

        self.convs = nn.ModuleList([Conv(in_chan, out_chan, **kwargs)
                                    for in_chan, out_chan in self.in_out_channels])

        self.norms = nn.ModuleList([Normalization(out_chan)
                                    for _, out_chan in self.in_out_channels])
        self.reset_parameters()

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
                 Pool=TopKPooling,
                 is_double_conv=False,
                 max_nchannels=512,
                 bottleneck=None,
                 n_layers=5,
                 is_force_same_bottleneck=False,
                 is_sum_res=True,
                 **kwargs):

        self.is_double_conv = is_double_conv
        self.max_nchannels = max_nchannels
        self.bottleneck = bottleneck
        self.is_sum_res = is_sum_res
        super().__init__(n_channels, Conv,
                         n_layers=n_layers,
                         is_res=False,
                         **kwargs)

        factor_chan = 1 if self.bottleneck == "channels" else 2
        pool_in_chan = [min(factor_chan**i * self.n_channels, self.max_nchannels)
                        for i in range(self.n_layers // 2 + 1)][1:]
        self.pools = nn.ModuleList([Pool(in_chan) for in_chan in pool_in_chan])
        self.is_force_same_bottleneck = is_force_same_bottleneck

        for pool in self.pools:
            pool.reset_parameters()

        self.reset_parameters()

    def apply_convs(self, X):

        x, edge_index, batch = X.x, X.edge_index, X.batch
        edge_weight = x.new_ones((edge_index.size(1), ))

        n_blocks = self.n_layers // 2 if self.is_double_conv else self.n_layers
        n_down_blocks = n_blocks // 2
        residuals = [None] * n_down_blocks
        edges = [None] * n_down_blocks
        perms = [None] * n_down_blocks

        # Down
        for i in range(n_down_blocks):
            x = self._apply_conv_block_i(x, edge_index, i, edge_weight=edge_weight)
            residuals[i] = x

            # not clear whether to save before or after augment
            edges[i] = (edge_index, edge_weight)

            # (A + I)^2
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))

            x, edge_index, edge_weight, batch, perm = self.pools[i](x, edge_index,
                                                                    edge_attr=edge_weight,
                                                                    batch=batch)
            perms[i] = perm

        # Bottleneck
        x = self._apply_conv_block_i(x, edge_index, n_down_blocks, edge_weight=edge_weight)
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
            edge_index, edge_weight = edges[n_down_blocks - i]
            res = residuals[n_down_blocks - i]
            up = torch.zeros_like(res)
            up[perms[n_down_blocks - i]] = x
            if not self.is_sum_res:
                x = torch.cat((res, up), dim=1)  # conncat on channels
            else:
                x = res + up
            x = self._apply_conv_block_i(x, edge_index, i, edge_weight=edge_weight)

        X.x = x

        return X, summary

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def _apply_conv_block_i(self, x, edge_index, i, **kwargs):
        """Apply the i^th convolution block."""
        if self.is_double_conv:
            i *= 2

        x = self.activation_(self.norms[i](self.convs[i](x, edge_index, **kwargs)))

        if self.is_double_conv:
            x = self.activation_(self.norms[i + 1](self.convs[i + 1](x, edge_index, **kwargs)))

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
            if not self.is_sum_res:
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
            if not self.is_sum_res:
                # e.g.: [(16, 32), (32,64), (64, 64), (128, 32), (64, 16)] due to concat
                idcs = slice(len(in_out_channels) // 2 + 1, len(in_out_channels))
                in_out_channels[idcs] = [(in_chan * 2, out_chan)
                                         for in_chan, out_chan in in_out_channels[idcs]]

        return in_out_channels


class GAT(nn.Module):
    def __init__(self, in_channels, out_channels=None, dim=8, heads=8, dropout=0.6, n_layers=2):
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

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_out.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index))

        x = self.conv_out(x, data.edge_index)
        data.x = x

        return data


def sort_edge_index(edge_index, edge_attr=None, num_nodes=None):
    r"""Row-wise sorts edge indices :obj:`edge_index`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index[0] * num_nodes + edge_index[1]
    perm = idx.argsort()

    return edge_index[:, perm], None if edge_attr is None else edge_attr[perm]


class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net graph
    architecture with graph pooling and unpooling operations.
    Args:
        channels (int): Size of each sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """

    def __init__(self, channels, depth, pool_ratios=0.5, sum_res=True,
                 act=F.relu):
        super(GraphUNet, self).__init__()

        self.channels = channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)

        self.act = act
        self.sum_res = sum_res

        self.initial_conv = GCNConv(channels, channels, improved=True)

        self.pools = torch.nn.ModuleList()
        self.down_convs = torch.nn.ModuleList()
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))
        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            in_channels = channels if sum_res else 2 * channels
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))

        self.final_conv = GCNConv(in_channels, channels, improved=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.initial_conv.reset_parameters()
        self.final_conv.reset_parameters()

        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.down_convs:
            conv.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, X):
        x, edge_index, batch = X.x, X.edge_index, X.batch

        x = self.initial_conv(x, edge_index)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(self.depth):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm = self.pools[i](
                x, edge_index, edge_weight, batch)
            x = self.act(self.down_convs[i](x, edge_index, edge_weight))

            if i < self.depth - 1:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            if i < self.depth - 1:
                x = self.act(self.up_convs[i](x, edge_index, edge_weight))

        x = self.final_conv(x, edge_index)

        X.x = x

        return X, None

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.channels, self.depth,
            self.pool_ratios)
