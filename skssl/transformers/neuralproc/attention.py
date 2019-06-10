import abc
import math
import torch

import torch.nn as nn
from torch.nn.modules.distance import CosineSimilarity, PairwiseDistance

from skssl.predefined import MLP
from skssl.utils.initialization import weights_init, linear_init
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
    elif attention == "multihead":
        attender = MultiheadAttender(kq_size, **kwargs)
    elif attention == "transformer":
        attender = TransformerAttender(kq_size, **kwargs)
    else:
        raise ValueError("Unknown attention method {}".format(attention))

    return attender


class BaseAttender(abc.ABC, nn.Module):
    """
    Base Attender module.

    Parameters
    ----------
    is_normalize : bool, optional
        Whether weights should sum to 1 (using softmax). If not weights will be
        in [0,1] but not necessarily sum to 1.

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
            attn = logits.sigmoid()
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
    Cosine similarity scorer for attention.
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
    Generalizes the Laplace (L1) attender from [1].

    Parameters
    ----------
    p : float, optional
        The exponent value in the norm formulation.

    length_scale : float, optional
        Float that scales down (divide) the distance.

    kwargs :
        Additional arguments to `BaseAttender`.

    References
    ----------
    [1] Kim, Hyunjik, et al. "Attentive neural processes." arXiv preprint
        arXiv:1901.05761 (2019).
    """

    def __init__(self, p=1, length_scale=1., **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.length_scale

    def score(self, keys, queries):
        batch_size, n_queries, kq_size = queries.size()
        n_keys = keys.size(1)

        keys = keys.view(batch_size, kq_size, 1, n_keys)
        queries = queries.view(batch_size, kq_size, n_queries, 1)

        logits = - F.normalize(keys - queries, p=self.p, dim=1) / self.length_scale

        return logits

    def logits_to_attn(self, logits):
        """Convert logits to attention."""
        if self.is_normalize:
            attn = logits.softmax(dim=-1)
        else:
            # changes the postprocessing as in
            # https://github.com/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb
            attn = logits.tanh() + 1
        return attn


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
        self.key_transform = nn.Linear(kqv_size, kqv_size, bias=False)
        self.query_transform = nn.Linear(kqv_size, kqv_size, bias=False)
        self.value_transform = nn.Linear(kqv_size, kqv_size, bias=False)
        self.dot = DotAttender(is_scale=True, dropout=dropout)
        self.n_heads = n_heads
        self.head_size = kqv_size // self.n_heads
        self.post_processor = nn.Linear(kqv_size, kqv_size) if is_post_process else None

        assert kqv_size % n_heads == 0
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

        # change initialization because no activation
        linear_init(self.key_transform, activation="linear")
        linear_init(self.query_transform, activation="linear")
        linear_init(self.value_transform, activation="linear")

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
        self.layer_norm = nn.LayerNorm(kqv_size)
        self.mlp = MLP(kqv_size, kqv_size)

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
        context = self.layer_norm(context + queries)
        context = self.layer_norm(context + self.dropout(self.mlp(context)))

        return context
