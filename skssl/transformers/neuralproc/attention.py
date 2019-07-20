import abc
import math
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.distance import CosineSimilarity, PairwiseDistance

from skssl.predefined import MLP
from skssl.utils.initialization import weights_init, init_param_
from skssl.utils.torchextend import identity


__all__ = ["get_attender"]


def get_attender(attention, kq_size=None, is_normalize=True, **kwargs):
    """
    Set scorer that matches key and query to compute attention along `dim=1`.

    Parameters
    ----------
    attention: {'multiplicative', "additive", "scaledot", "multihead", "manhattan",
             "euclidean", "cosine"}, optional
        The method to compute the alignment. `"scaledot"` mitigates the high
        dimensional issue of the scaled product by rescaling it [1]. `"multihead"`
        is the same with multiple heads [1]. `"additive"` is the original attention
        [2]. `"multiplicative"` is faster and more space efficient [3]
        but performs a little bit worst for high dimensions. `"cosine"` cosine
        similarity. `"manhattan"` `"euclidean"` are the negative distances.

    kq_size : int, optional
        Size of the key and query. Only needed for 'multiplicative', 'additive'
        "multihead".

    is_normalize : bool, optional
        Whether qttention weights should sum to 1 (using softmax). If not weights
        will be in [0,1] but not necessarily sum to 1.

    kwargs :
        Additional arguments to the attender.

    References
    ----------
    [1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural
        information processing systems. 2017.
    [2] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine
        translation by jointly learning to align and translate." arXiv preprint
        arXiv:1409.0473 (2014).
    [3] Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "Effective
        approaches to attention-based neural machine translation." arXiv preprint
        arXiv:1508.04025 (2015).
    """
    attention = attention.lower()
    if attention == 'multiplicative':
        attender = MultiplicativeAttender(kq_size, is_normalize=is_normalize, **kwargs)
    elif attention == 'additive':
        attender = AdditiveAttender(kq_size, is_normalize=is_normalize, **kwargs)
    elif attention == 'scaledot':
        attender = DotAttender(is_scale=True, is_normalize=is_normalize, **kwargs)
    elif attention == "cosine":
        attender = CosineAttender(is_normalize=is_normalize, **kwargs)
    elif attention == "manhattan":
        attender = DistanceAttender(p=1, is_normalize=is_normalize, **kwargs)
    elif attention == "euclidean":
        attender = DistanceAttender(p=2, is_normalize=is_normalize, **kwargs)
    elif attention == "weighted_dist":
        attender = DistanceAttender(kq_size=kq_size, is_weight=True, p=1,
                                    is_normalize=is_normalize, **kwargs)
    elif attention == "multihead":
        attender = MultiheadAttender(kq_size, **kwargs)
    elif attention == "transformer":
        attender = TransformerAttender(kq_size, **kwargs)
    elif attention == "generalized_conv":
        attender = GeneralizedConvAttender(1, is_normalize=is_normalize, **kwargs)
    else:
        raise ValueError("Unknown attention method {}".format(attention))

    return attender


class BaseAttender(abc.ABC, nn.Module):
    """
    Base Attender module.

    Parameters
    ----------
    is_normalize : bool, optional
        Whether weights should sum to 1 (using softmax). If not weights will not
        be normalized.

    dropout : float, optional
        Dropout rate to apply to the attention.
    """

    def __init__(self, is_normalize=True, dropout=0):
        super().__init__()
        self.is_normalize = is_normalize
        self.dropout = (nn.Dropout(p=dropout) if dropout > 0 else identity)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, keys, queries, values, **kwargs):
        """
        Compute the attention between given key and queries.

        Parameters
        ----------
        keys : torch.Tensor, size=[batch_size, n_keys, kq_size]
        queries : torch.Tensor, size=[batch_size, n_queries, kq_size]
        values : torch.Tensor, size=[batch_size, n_keys, value_size]

        Return
        ------
        context : torch.Tensor, size=[batch_size, n_queries, value_size]
        """
        logits = self.score(keys, queries, **kwargs)

        attn = self.logits_to_attn(logits)

        attn = self.dropout(attn)

        # attn : size=[batch_size, n_queries, n_keys]
        # values : size=[batch_size, n_keys, value_size]
        context = torch.bmm(attn, values)

        return context

    def logits_to_attn(self, logits):
        """Convert logits to attention."""
        if self.is_normalize:
            attn = logits.softmax(dim=-1)
        else:
            attn = logits
        return attn

    @abc.abstractmethod
    def score(keys, queries, **kwargs):
        """Score function which returns the logits between keys and queries."""
        pass


class DotAttender(BaseAttender):
    """
    Dot product attention.

    Parameters
    ----------
    is_scale: bool, optional
        whether to use a scaled attention just like in [1]. Scaling can help when
        dimension is large by making sure that there are no extremely small gradients.

    kwargs:
        Additional arguments to `BaseAttender`.

    References
    ----------
    [1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural
        information processing systems. 2017.
    """

    def __init__(self, is_scale=True, **kwargs):
        super().__init__(**kwargs)
        self.is_scale = is_scale

    def score(self, keys, queries):
        # [batch_size, n_queries, kq_size] * [batch_size, kq_size, n_keys]
        logits = torch.bmm(queries, keys.transpose(1, 2))

        if self.is_scale:
            kq_size = queries.size(-1)
            logits = logits / math.sqrt(kq_size)

        return logits


class MultiplicativeAttender(BaseAttender):
    """
    Multiplicative attention mechanism [1].

    Parameters
    ----------
    kq_size: int
        Size of key and query.

    kwargs:
        Additional arguments to `BaseAttender`.

    References
    ----------
    [1] Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "Effective
        approaches to attention-based neural machine translation." arXiv preprint
        arXiv:1508.04025 (2015).
    """

    def __init__(self, kq_size, **kwargs):
        super().__init__(**kwargs)

        self.linear = nn.Linear(kq_size, kq_size, bias=False)
        self.dot = DotAttender(is_scale=False)
        self.reset_parameters()

    def score(self, keys, queries):
        transformed_queries = self.linear(queries)
        logits = self.dot.score(keys, transformed_queries)
        return logits


class AdditiveAttender(BaseAttender):
    """
    Original additive attention mechanism [1].

    Parameters
    ----------
    kq_size: int
        Size of key and query.

    kwargs:
        Additional arguments to `BaseAttender`.

    References
    ----------
    [1] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine
        translation by jointly learning to align and translate." arXiv preprint
        arXiv:1409.0473 (2014).
    """

    def __init__(self, kq_size, **kwargs):
        super().__init__(**kwargs)

        self.mlp = MLP(kq_size * 2, 1, hidden_size=kq_size, activation=nn.Tanh)
        self.reset_parameters()

    def score(self, keys, queries):
        batch_size, n_queries, kq_size = queries.shape
        n_keys = keys.size(1)

        keys = keys.unsqueeze(1).expand(batch_size, n_queries, n_keys, kq_size)
        queries = queries.unsqueeze(1).expand(batch_size, n_queries, n_keys, kq_size)

        logits = self.mlp(torch.cat((keys, queries), dim=-1)).squeeze(-1)
        return logits


class CosineAttender(BaseAttender):
    """
    Computes the attention as a function of cosine similarity.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.similarity = CosineSimilarity(dim=1)

    def score(self, keys, queries):
        batch_size, n_queries, kq_size = queries.size()
        n_keys = keys.size(1)

        keys = keys.view(batch_size, kq_size, 1, n_keys)
        queries = queries.view(batch_size, kq_size, n_queries, 1)
        logits = self.similarity(keys, queries)

        return logits


class DistanceAttender(BaseAttender):
    """
    Computes the attention as a function of the negative dimension wise (weighted)
    distance.

    Parameters
    ----------
    p : float, optional
        The exponent value in the norm formulation.

    is_weight : float, optional
        Whether to use a dimension wise weight and bias.

    kwargs :
        Additional arguments to `BaseAttender`.
    """

    def __init__(self, kq_size=None, p=1, is_weight=False, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.is_weight = is_weight
        if self.is_weight:
            self.weighter = nn.Linear(kq_size, kq_size)

        self.reset_parameters()

    def score(self, keys, queries):
        batch_size, n_queries, kq_size = queries.size()
        n_keys = keys.size(1)

        keys = keys.view(batch_size, 1, n_keys, kq_size)
        queries = queries.view(batch_size, n_queries, 1, kq_size)
        diff = keys - queries
        if self.is_weight:
            diff = self.weighter(diff)

        logits = - torch.norm(diff, p=self.p, dim=-1)

        return logits


class MultiheadAttender(nn.Module):
    """
    Multihead attention mechanism [1].

    Parameters
    ----------
    kq_size: int
        Size of key and query.

    n_heads : int, optional
        Number of heads

    is_post_process : bool, optional
        Whether to pos process the outout with a linear layer.

    dropout : float, optional
        Dropout rate to apply to the attention.

    References
    ----------
    [1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural
        information processing systems. 2017.
    """

    def __init__(self, kqv_size, n_heads=8, is_post_process=True, dropout=0):
        super().__init__()
        # only 3 transforms for scalability but actually as if using n_heads * 3 layers
        self.key_transform = nn.Linear(kqv_size, kqv_size, bias=False)
        self.query_transform = nn.Linear(kqv_size, kqv_size, bias=False)
        self.value_transform = nn.Linear(kqv_size, kqv_size, bias=False)
        self.dot = DotAttender(is_scale=True, dropout=dropout)
        self.n_heads = n_heads
        self.head_size = kqv_size // self.n_heads
        self.kqv_size = kqv_size
        self.post_processor = nn.Linear(kqv_size, kqv_size) if is_post_process else None

        assert kqv_size % n_heads == 0
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

        # change initialization because real output is not kqv_size but head_size
        # just coded so for convenience and scalability
        std = math.sqrt(2.0 / (self.kqv_size + self.head_size))
        nn.init.normal_(self.key_transform.weight, mean=0, std=std)
        nn.init.normal_(self.query_transform.weight, mean=0, std=std)
        nn.init.normal_(self.value_transform.weight, mean=0, std=std)

    def forward(self, keys, queries, values, **kwargs):
        """
        Compute the attention between given key and queries.

        Parameters
        ----------
        keys: torch.Tensor, size=[batch_size, n_keys, kqv_size]
        queries: torch.Tensor, size=[batch_size, n_queries, kqv_size]
        values: torch.Tensor, size=[batch_size, n_keys, kqv_size]

        Return
        ------
        context : torch.Tensor, size=[batch_size, n_queries, kqv_size]
        """
        keys = self.key_transform(keys)
        queries = self.query_transform(queries)
        values = self.value_transform(values)

        # Make multihead. Size = [batch_size * n_heads, {n_keys, n_queries}, head_size]
        keys = self._make_multiheaded(keys)
        values = self._make_multiheaded(values)
        queries = self._make_multiheaded(queries)

        # [batch_size * n_heads, n_queries, head_size]
        context = self.dot(keys, queries, values)

        context = self._concatenate_multiheads(context)

        if self.post_processor is not None:
            context = self.post_processor(context)

        return context

    def _make_multiheaded(self, kvq):
        """Make a key, value, query multiheaded by stacking the heads as new batches."""
        batch_size = kvq.size(0)
        kvq = kvq.view(batch_size, -1, self.n_heads, self.head_size)
        kvq = kvq.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads,
                                                        -1,
                                                        self.head_size)
        return kvq

    def _concatenate_multiheads(self, kvq):
        """Reverts `_make_multiheaded` by concatenating the heads."""
        batch_size = kvq.size(0) // self.n_heads
        kvq = kvq.view(self.n_heads, batch_size, -1, self.head_size)
        kvq = kvq.permute(1, 2, 0, 3).contiguous().view(batch_size,
                                                        -1,
                                                        self.n_heads * self.head_size)
        return kvq


class TransformerAttender(MultiheadAttender):
    """
    Image Transformer attention mechanism [1].

    Parameters
    ----------
    kq_size: int
        Size of key and query.

    kwargs:
        Additional arguments to `MultiheadAttender`.

    References
    ----------
    [1] Parmar, Niki, et al. "Image transformer." arXiv preprint arXiv:1802.05751
        (2018).
    """

    def __init__(self, kqv_size, **kwargs):
        super().__init__(kqv_size, is_post_process=False, **kwargs)
        self.layer_norm1 = nn.LayerNorm(kqv_size)
        self.layer_norm2 = nn.LayerNorm(kqv_size)
        self.mlp = MLP(kqv_size, kqv_size, hidden_size=kqv_size, activation=nn.ReLU)

        self.reset_parameters()

    def forward(self, keys, queries, values, **kwargs):
        """
        Compute the attention between given key and queries.

        Parameters
        ----------
        keys: torch.Tensor, size=[batch_size, n_keys, kqv_size]
        queries: torch.Tensor, size=[batch_size, n_queries, kqv_size]
        values: torch.Tensor, size=[batch_size, n_keys, kqv_size]

        Return
        ------
        context : torch.Tensor, size=[batch_size, n_queries, kqv_size]
        """
        context = super().forward(keys, queries, values)
        # residual connection + layer norm
        context = self.layer_norm1(context + queries)
        context = self.layer_norm2(context + self.dot.dropout(self.mlp(context)))

        return context


class GaussianRBF(nn.Module):
    def __init__(self):
        super().__init__()
        self.length_scale = nn.Parameter(torch.tensor([-1.]))
        self.reset_parameters()

    def reset_parameters(self):
        # initial length scale of ~0.3 => assumes that does "3" variations
        self.length_scale = nn.Parameter(torch.tensor([-1.]))

    def forward(self, x):
        out = torch.exp(- (x / F.softplus(self.length_scale)).pow(2))
        return out


class BroadcastedConv(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(1, 1, *args, **kwargs)
        self.linear = nn.Linear(self.in_channels, self.out_channels)
        self.reset_parameters()

    def forward(self, x):
        batch_size = x.size(0)
        # put in channels as batches
        x = x.view(batch_size * self.in_channels, 1, -1)
        x = self.conv(x)
        x = x.view(batch_size, -1, self.in_channels)
        x = self.linear(x)
        x = x.view(batch_size, self.out_channels, -1)
        return x

    def reset_parameters(self):
        weights_init(self)


class DepthSepConv(nn.Module):
    def __init__(self, nin, nout, *args, kernels_per_layer=1, **kwargs):
        super().__init__()
        self.kernels_per_layer = nin if kernels_per_layer is None else kernels_per_layer
        self.depthwise = nn.Conv1d(nin, nin * self.kernels_per_layer, *args,
                                   groups=nin, **kwargs)
        self.pointwise = nn.Conv1d(nin * self.kernels_per_layer, nout, 1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

    def reset_parameters(self):
        weights_init(self)


class GeneralizedConvAttender(nn.Module):
    """WIP"""

    def __init__(self, value_size,
                 RadialFunc=GaussianRBF,
                 n_tmp_queries=1024,
                 n_conv=5,
                 activation=nn.ReLU,
                 is_batchnorm=False,
                 kernel_size=7,
                 is_normalize=True,
                 is_concat=True,
                 n_chan=30,
                 is_depth_separable_conv=True,
                 dilation=2,
                 is_u_style=False,
                 **kwargs):
        super().__init__()
        self.value_size = value_size
        # add density channel
        n_input_chan = self.value_size + 1
        self.radial_func = RadialFunc()
        self.is_normalize = is_normalize
        self.n_chan = n_chan
        self.n_conv = n_conv
        self.is_u_style = is_u_style
        self.is_concat = is_concat
        self.n_tmp_queries = n_tmp_queries
        self.activation = activation()

        # linear layer to mix channels
        self.linear = nn.Linear(n_input_chan, n_chan)
        # density layer transform
        self.density_transform = nn.Linear(1, 1)
        conv = DepthSepConv if is_depth_separable_conv else nn.Conv1d

        self.convs = nn.ModuleList([conv(self.n_chan, self.n_chan, kernel_size,
                                         padding=(kernel_size // 2) * dilation,
                                         dilation=dilation)
                                    for _ in range(self.n_conv)])

        if self.n_conv > 0:
            self.tmp_queries = torch.linspace(-1, 1, self.n_tmp_queries)

        normalization = nn.BatchNorm1d if is_batchnorm else nn.Identity
        self.norms = nn.ModuleList([normalization(self.n_chan) for _ in range(self.n_conv)])

        if self.is_u_style:
            self.convs_up = nn.ModuleList([conv(self.n_chan * 2, self.n_chan, kernel_size,
                                                padding=(kernel_size // 2) * dilation,
                                                dilation=dilation)
                                           for _ in range(self.n_conv)])
            self.norms_up = nn.ModuleList([normalization(self.n_chan)
                                           for _ in range(self.n_conv * 2)])

        inp = self.n_chan + n_input_chan if self.is_concat else self.n_chan
        # 2 * size because mean and variance
        self.pred = MLP(inp, 2 * value_size, hidden_size=16, n_hidden_layers=3)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def extend_tmp_queries(self, min_max):
        """
        Scale the temporary queries to be in a given range while keeping
        the same density than during training (used for extrapolation.).
        """
        # reset
        self.tmp_queries = torch.linspace(-1, 1, self.n_tmp_queries)

        current_min = min(self.tmp_queries)
        current_max = max(self.tmp_queries)
        delta = self.tmp_queries[1] - self.tmp_queries[0]
        n_queries_per_increment = len(self.tmp_queries) / (current_max - current_min)
        n_add_left = math.ceil((current_min - min_max[0]) * n_queries_per_increment)
        n_add_right = math.ceil((min_max[1] - current_max) * n_queries_per_increment)

        tmp_queries = []
        if n_add_left > 0:
            tmp_queries.append(torch.arange(min_max[0], current_min, delta))
        tmp_queries.append(self.tmp_queries)
        if n_add_right > 0:
            # add delta to not have twice the previous max boundary
            tmp_queries.append(torch.arange(current_max, min_max[1], delta) + delta)
        self.tmp_queries = torch.cat(tmp_queries)

    def forward(self, keys, queries, values, **kwargs):
        """
        Compute the attention between given key and queries.

        Parameters
        ----------
        keys : torch.Tensor, size=[batch_size, n_keys, kq_size]
        queries : torch.Tensor, size=[batch_size, n_queries, kq_size]
        values : torch.Tensor, size=[batch_size, n_keys, value_size]

        Return
        ------
        context : torch.Tensor, size=[batch_size, n_queries, value_size]
        """
        batch_size, n_keys, value_size = values.shape
        _, n_queries, kq_size = queries.shape
        assert kq_size == 1

        keys = keys.view(batch_size, 1, n_keys, kq_size)
        values = values.view(batch_size, 1, n_keys, value_size)
        queries = queries.view(batch_size, n_queries, 1, kq_size)

        if self.n_conv != 0:
            self.tmp_queries = self.tmp_queries.to(values.device)
            tmp_queries = self.tmp_queries.view(1, -1, 1, 1)
            # batch_size, n_tmp_queries, value_size + 1
            grid_values = self.non_grided_conv(keys, tmp_queries, values,
                                               is_concat_density=True)
            grid_values = self.linear(grid_values)
            grid_values = self.grided_conv(keys, tmp_queries, queries, grid_values)

            tmp_queries = self.tmp_queries.view(1, 1, -1, 1)
            out = self.non_grided_conv(tmp_queries, queries, grid_values)
        else:
            out = self.non_grided_conv(keys, queries, values, is_concat_density=True)
            out = self.activation(self.linear(out))

        if self.is_concat:
            # concateate the density channel and direct predictions
            to_concat = self.non_grided_conv(keys, queries, values,
                                             is_concat_density=True)
            out = torch.cat([out, to_concat], dim=-1)

        return self.pred(out)

    def non_grided_conv(self, keys, queries, values, is_concat_density=False):
        # batch_size, n_queries, n_keys, 1
        dist = torch.norm(keys - queries, p=2, dim=-1, keepdim=True)
        weight = self.radial_func(dist)
        if is_concat_density:
            density_chan = - weight.sum(dim=2)

        if self.is_normalize:
            weight = torch.nn.functional.normalize(weight, dim=2, p=1)
        elif is_concat_density:
            weight = weight / 40
        else:
            # equivalent to changing initialization of convs
            weight = weight / len(self.tmp_queries)

        # batch_size, n_queries, value_size
        values = (weight * values).sum(dim=2)

        if is_concat_density:
            density_chan = torch.sigmoid(self.density_transform(density_chan))
            # don't normalize the density channel
            values = torch.cat([values, density_chan], dim=-1)

        return values

    def grided_conv(self, keys, tmp_queries, queries, grid_values):
        batch_size = grid_values.size(0)
        grid_values = grid_values.view(batch_size, self.n_chan, -1)

        if self.is_u_style:
            # convs down
            grid_values_down = [None] * len(self.convs)
            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                grid_values_down[i] = grid_values
                grid_values = F.interpolate(grid_values,
                                            mode="linear",
                                            scale_factor=0.5,
                                            align_corners=True)
                # normalization and residual
                grid_values = conv(self.activation(norm(grid_values)))

            # convs up
            for i, (conv, norm) in enumerate(zip(self.convs_up, self.norms_up)):
                grid_values = F.interpolate(grid_values,
                                            mode="linear",
                                            scale_factor=2,
                                            align_corners=True)
                # concat unet style on chanel
                grid_values = torch.cat([grid_values, grid_values_down[-i - 1]], dim=-2)
                # normalization and residual
                grid_values = conv(self.activation(norm(grid_values)))

        else:
            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                # normalization and residual
                grid_values = conv(self.activation(norm(grid_values))) + grid_values

        # batch_size, 1, n_tmp_queries, 1, channel
        grid_values = grid_values.view(batch_size, 1, -1, self.n_chan)

        return grid_values
