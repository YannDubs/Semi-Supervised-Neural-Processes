
import abc
import math
import torch

import torch.nn as nn
from torch.nn.modules.distance import CosineSimilarity, PairwiseDistance

from skssl.predefined import MLP
from skssl.utils.initialization import weights_init


def get_attender(scorer, kq_size, is_normalize=True):
    """
    Set scorer that matches key and query to compute attention along `dim=1`.

    scorer: {'multiplicative', "additive", "scaledot", "multihead", "manhattan",
             "euclidean", "cosine"}, optional
        The method to compute the alignment. `"scaledot"` [Vaswani et al., 2017]
        mitigates the high dimensional issue by rescaling the dot product.
        `"multihead"` is th esame with multiple heads.
        `"additive"` is the original  attention [Bahdanau et al., 2015].
        `"multiplicative"` is faster and more space efficient [Luong et al., 2015]
        but performs a little bit worst for high dimensions. `"cosine"` cosine
        similarity. `"manhattan"` `"euclidean"` are the negative distances.
    """
    scorer = scorer.lower()
    if scorer == 'multiplicative':
        scorer = MultiplicativeAttender(kq_size, is_normalize=is_normalize)
    elif scorer == 'additive':
        scorer = AdditiveAttender(kq_size, is_normalize=is_normalize)
    elif scorer == 'scaledot':
        scorer = DotAttender(is_scale=True, is_normalize=is_normalize)
    elif scorer == "cosine":
        scorer = CosineAttender(is_normalize=is_normalize)
    elif scorer == "manhattan":
        scorer = DistanceAttender(p=1, is_normalize=is_normalize)
    elif scorer == "euclidean":
        scorer = DistanceAttender(p=2, is_normalize=is_normalize)
    else:
        raise ValueError("Unknown attention method {}".format(scorer))

    return scorer


class BaseAttender(abc.ABC, nn.Module):
    """
    Base Attender module.

    is_normalize: bool, optional
        Whether weights should sum to 1 (using softmax). If not weights will be
        in [0,1] but not necessarily sum to 1.
    """

    def __init__(self, is_normalize=True):
        super().__init__()
        self.is_normalize = is_normalize
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, keys, queries, values, **kwargs):
        """
        Compute the attention between given key and queries.

        Parameters
        ----------
        keys: torch.Tensor, size=[batch_size, n_keys, kq_size]
        queries: torch.Tensor, size=[batch_size, n_queries, kq_size]
        values: torch.Tensor, size=[batch_size, n_keys, value_size]

        Return
        ------
        glimpse: torch.Tensor, size=[batch_size, n_queries, value_size]
        """
        logits = self.score(keys, queries, **kwargs)

        attn = self.logits_to_attn(logits)

        # attn : size=[batch_size, n_queries, n_keys]
        # values : size=[batch_size, n_keys, value_size]
        glimpse = torch.bmm(attn, values)

        return glimpse

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

    Notes
    -----
    - Only difference with paper is that adds bias.

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
        self.linear = nn.Linear(kq_size, kq_size)
        self.dot = DotAttender(is_scale=False)

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
