import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.independent import Independent
from torch.distributions import Normal


from skssl.utils.initialization import weights_init, init_param_
from skssl.utils.torchextend import min_max_scale, MultivariateNormalDiag
from skssl.predefined import (merge_flat_input, discard_ith_arg, SetConv, GaussianRBF,
                              RelativeSinusoidalEncodings, MLP, UnetCNN, get_attender)

from .datasplit import CntxtTrgtGetter


__all__ = ["NeuralProcess", "AttentiveNeuralProcess", "GlobalNeuralProcess"]


class NeuralProcess(nn.Module):
    """
    Implements (Conditional [2]) Neural Process [1] using tricks from [3] for
    functions of arbitrary dimensions.

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    r_dim : int, optional
        Dimension of representation.

    x_transf_dim : int, optonal
        Dimension of the encoded X. If `-1` uses `r_dim`. if `None` uses `x_dim`.

    XEncoder : nn.Module, optional
        Spatial encoder module which maps {x_i} -> x_transf_i. It should be
        constructable via `xencoder(x_dim, x_transf_dim)`. Example:
            - `MLP` : will learn positional embeddings with MLP
            - `SinusoidalEncodings` : use sinusoidal positional encodings.

    XYEncoder : nn.Module, optional
        Encoder module which maps {x_transf_i, y_i} -> {r_i}. It should be constructable
        via `xyencoder(x_transf_dim, y_dim, n_out)`. If you have an encoder that maps
        xy -> r you can convert it via `merge_flat_input(Encoder)`. `None` uses
        parameter dependent default. Example:
            - `merge_flat_input(MLP, is_sum_merge=False)` : learn representation
            with MLP. `merge_flat_input` concatenates (or sums) X and Y inputs.
            - `merge_flat_input(SelfAttention, is_sum_merge=True)` : self attention
            mechanisms as [4]. For more parameters (attention type, number of
            layers ...) refer to its docstrings.
            - `discard_ith_arg(MLP, 0)` if want the encoding to only depend on Y.

    Decoder : nn.Module, optional
        Decoder module which maps {x_t, r} -> {y_hat_t}. It should be constructable
        via `decoder(x, r_dim, n_out)`. If you have an decoder that maps
        rx -> y you can convert it via `merge_flat_input(Decoder)`. `None` uses
        parameter dependent default. Example:
            - `merge_flat_input(MLP)` : predict with MLP.
            - `merge_flat_input(SelfAttention, is_sum_merge=True)` : predict
            with self attention mechanisms (using `X_transf + Y` as input) to have
            coherant predictions (not use in attentive neural process [4] but in
            image transformer [5]).
            - `discard_ith_arg(MLP, 0)` if want the decoding to only depend on r.

    aggregator : callable, optional
        Agregreator function which maps {r_i} -> r. It should have a an argument
        `dim` to say specify the dimensions of aggregation. The dimension should
        not be kept (i.e. keepdim=False). To use a cross attention aggregation,
        use `AttentiveNeuralProcess` instead of `NeuralProcess`.

    LatentEncoder : nn.Module, optional
        Encoder which maps r -> z_suff_stat. It should be constructed via
        `LatentEncoder(r_dim, n_out)`. Only used if `encoded_path in ["latent",
        "both"]`.

    get_cntxt_trgt : callable, optional
        Function that split the input into context and target points.
        `X_cntxt, Y_cntxt, X_trgt, Y_trgt = self.get_cntxt_trgt(X, y, **kwargs)`.
        Note: context points should be a subset of target ones. If you already
        have the context and target point, put them in a dictionary and split
        the dictionary in `get_cntxt_trgt`.

    encoded_path : {"deterministic", "latent", "both"}
        Which path(s) to use:
        - `"deterministic"` uses a Conditional Neural Process [2] (no latents),
        where the decoder gets a deterministic representation as input
        (function of the context).
        - `"latent"` uses the original Neural Process [1], where the decoder gets
        a sample latent representation as input (function of the target during
        training and context during test).
        If `"both"` concatenates both representations as described in [4].

    PredictiveDistribution : torch.distributions.Distribution, optional
        Predictive distribution. The predicted outputs will be independent and thus
        wrapped around `Independent` (e.g. diagonal covariance for a Gaussian).
        The input to the constructor are currently a value in ]-inf, inf[ and one
        in [0.1, inf[ (typically `loc` and `scale`), although it is very easy to make
        more general if needs be.

    is_use_x : bool, optional
        Whether to encode and use X in the representation (r_i) and when decoding.
        If `False`, then guarantees translation equivariance (if add some
        representation of the positional differences) or invariance.

    References
    ----------
    [1] Garnelo, Marta, et al. "Neural processes." arXiv preprint
        arXiv:1807.01622 (2018).
    [2] Garnelo, Marta, et al. "Conditional neural processes." arXiv preprint
        arXiv:1807.01613 (2018).
    [3] Le, Tuan Anh, et al. "Empirical Evaluation of Neural Process Objectives."
        NeurIPS workshop on Bayesian Deep Learning. 2018.
    [4] Kim, Hyunjik, et al. "Attentive neural processes." arXiv preprint
        arXiv:1901.05761 (2019).
    [5] Parmar, Niki, et al. "Image transformer." arXiv preprint arXiv:1802.05751
        (2018).
    """

    def __init__(self, x_dim, y_dim,
                 r_dim=128,
                 x_transf_dim=-1,
                 XEncoder=MLP,
                 XYEncoder=None,
                 Decoder=None,
                 aggregator=torch.mean,
                 LatentEncoder=MLP,
                 get_cntxt_trgt=CntxtTrgtGetter(is_add_cntxts_to_trgts=True),
                 encoded_path="deterministic",
                 PredictiveDistribution=Normal,
                 is_use_x=True,
                 min_std=0.1,
                 Classifier=None):
        super().__init__()

        Decoder, XYEncoder, x_transf_dim, XEncoder = self._get_defaults(Decoder,
                                                                        XYEncoder,
                                                                        x_transf_dim,
                                                                        XEncoder,
                                                                        is_use_x,
                                                                        r_dim)

        self.min_std = min_std
        self.get_cntxt_trgt = get_cntxt_trgt
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.encoded_path = encoded_path.lower()
        self.PredictiveDistribution = PredictiveDistribution
        self.is_transform = False
        self.classifier = Classifier()

        if x_transf_dim is None:
            self.x_transf_dim = self.x_dim
        elif x_transf_dim == -1:
            self.x_transf_dim = self.r_dim
        else:
            self.x_transf_dim = x_transf_dim

        self.x_encoder = XEncoder(self.x_dim, self.x_transf_dim)
        self.xy_encoder = XYEncoder(self.x_transf_dim, self.y_dim, self.r_dim)
        self.aggregator = aggregator
        # *2 because mean and var
        self.decoder = Decoder(self.x_transf_dim, self.r_dim, self.y_dim * 2)

        if self.encoded_path in ["latent", "both"]:
            self.lat_encoder = LatentEncoder(self.r_dim, self.r_dim * 2)
            if self.encoded_path == "both":
                self.merge_rz = nn.Linear(self.r_dim * 2, self.r_dim)
        elif self.encoded_path == "deterministic":
            self.lat_encoder = None
        else:
            raise ValueError("Unkown encoded_path={}.".format(encoded_path))

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def _get_defaults(self, Decoder, XYEncoder, x_transf_dim, XEncoder, is_use_x, r_dim):
        # don't use `x` to be translation equivariant
        dflt_sub_decoder = partial(MLP, n_hidden_layers=4, is_force_hid_smaller=True)
        dflt_sub_xyencoder = partial(MLP, n_hidden_layers=2, is_force_hid_smaller=True)

        if not is_use_x:
            if Decoder is None:
                Decoder = discard_ith_arg(dflt_sub_decoder, i=0)  # depend only on r not x

            if XYEncoder is None:
                XYEncoder = discard_ith_arg(dflt_sub_xyencoder, i=0)  # depend only on y not x

            x_transf_dim = None  # don't encode X
            XEncoder = nn.Identity  # don't encode X
        else:
            if Decoder is None:
                Decoder = merge_flat_input(dflt_sub_decoder, is_sum_merge=True)
            if XYEncoder is None:
                XYEncoder = merge_flat_input(dflt_sub_xyencoder, is_sum_merge=True)

        return Decoder, XYEncoder, x_transf_dim, XEncoder

    def forward(self, X, y=None, **kwargs):
        """
        Split context and target in the class to make it compatible with
        usual datasets and training frameworks, then redirects to `forward_step`.
        """

        try:
            X_cntxt, Y_cntxt, X_trgt, Y_trgt = self.get_cntxt_trgt(X, y, **kwargs)
        except:
            import pdb
            pdb.set_trace()

        # fro now onwards, all inputs are assumed to be in [-1,1] when training
        if self.training:
            if X_cntxt.min() < -1 or X_trgt.min() < -1:
                raise ValueError("Position inputs during training should be in [-1,1] (besides sparse dim). {} < X_cntxt  ; {} < X_trgt .".format(X_cntxt.min(), X_trgt.min()))

        return self.forward_step(X_cntxt, Y_cntxt, X_trgt, Y_trgt=Y_trgt)

    def forward_step(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        """
        Given a set of context pairs {x_i, y_i} and target points {x_t}, return
        a set of posterior distribution over target points {y_trgt}.

        Parameters
        ----------
        X_cntxt: torch.Tensor, size=[batch_size, n_cntxt, x_dim]
            Set of all context features {x_i}.

        Y_cntxt: torch.Tensor, size=[batch_size, n_cntxt, y_dim]
            Set of all context values {y_i}.

        X_trgt: torch.Tensor, size=[batch_size, n_trgt, x_dim]
            Set of all target features {x_t}. Note: context points should be a
            subset of target ones.

        Y_trgt: torch.Tensor, size=[batch_size, n_trgt, y_dim], optional
            Set of all target values {y_t}. Only required during training.

        Return
        ------
        p_y_trgt: torch.distributions.Distribution
            Target distribution.

        Y_trgt: torch.Tensor, size=[batch_size, n_trgt, y_dim]
            Set of all target values {y_t}, returned to redirect it to the loss
            function.

        q_z_trgt: torch.distributions.Distribution
            Latent distribution for the targets. `None` if `LatentEncoder=None`
            or not training.

        q_z_cntxt: torch.distributions.Distribution
            Latent distribution for the context points. `None` if
            `LatentEncoder=None` or not training.
        """

        R_det, z_sample, q_z_cntxt, q_z_trgt = None, None, None, None

        if self.encoded_path in ["latent", "both"]:
            z_sample, q_z_cntxt = self.latent_path(X_cntxt, Y_cntxt)

            if self.training:
                # during training when we know Y_trgt, we compute the latent using
                # the targets as context (which also contain the context). If we
                # used it for the deterministic path, then the model would cheat
                # by learning a point representation for each function => bad representation
                z_sample, q_z_trgt = self.latent_path(X_trgt, Y_trgt)

        if self.is_transform and not self.training:
            representation = self.deterministic_path(X_cntxt, Y_cntxt, None)
            # for transform you want representation (could also want mean_z but r
            # should have this info).
            return representation

        if self.encoded_path in ["deterministic", "both"]:
            R_det, summary = self.deterministic_path(X_cntxt, Y_cntxt, X_trgt)

        dec_inp = self.make_dec_inp(R_det, z_sample, X_trgt)
        p_y_trgt = self.decode(dec_inp, X_trgt)

        if self.classifier is not None:
            pred_logits = self.classifier(summary)
            return pred_logits, p_y_trgt, Y_trgt, q_z_trgt, q_z_cntxt, None, X_trgt

        if not self.training:
            summary = None  # summary None =Y don't use in loss

        return p_y_trgt, Y_trgt, q_z_trgt, q_z_cntxt, summary, X_trgt

    def latent_path(self, X, Y):
        """Latent encoding path."""
        # size = [batch_size, n_cntxt, x_transf_dim]
        X_transf = self.x_encoder(X)
        # size = [batch_size, n_cntxt, x_transf_dim]
        R_cntxt = self.xy_encoder(X_transf, Y)

        # size = [batch_size, r_dim]
        r = self.aggregator(R_cntxt, dim=1)

        z_suff_stat = self.lat_encoder(r)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives".
        mean_z, std_z = z_suff_stat.view(z_suff_stat.shape[0], -1, 2).unbind(-1)
        std_z = 0.1 + 0.9 * torch.sigmoid(std_z)
        # use a Gaussian prior on latent
        q_z = MultivariateNormalDiag(mean_z, std_z)
        z_sample = q_z.rsample()

        return z_sample, q_z

    def deterministic_path(self, X_cntxt, Y_cntxt, X_trgt):
        """
        Deterministic encoding path. `X_trgt` can be used in child classes
        to give a target specific representation (e.g. attentive neural processes).
        """
        # size = [batch_size, n_cntxt, x_transf_dim]
        X_transf = self.x_encoder(X_cntxt)
        # size = [batch_size, n_cntxt, x_transf_dim]
        R_cntxt = self.xy_encoder(X_transf, Y_cntxt)

        # size = [batch_size, r_dim]
        r = self.aggregator(R_cntxt, dim=1)

        if X_trgt is None:
            return r

        batch_size, n_trgt, _ = X_trgt.shape
        R = r.unsqueeze(1).expand(batch_size, n_trgt, self.r_dim)
        return R, None

    def make_dec_inp(self, R, z_sample, X_trgt):
        """Make the context input for the decoder."""
        batch_size, n_trgt, _ = X_trgt.shape

        if self.encoded_path == "both":
            Z = z_sample.unsqueeze(1).expand(batch_size, n_trgt, self.r_dim)
            dec_inp = torch.relu(self.merge_rz(torch.cat((R, Z), dim=-1)))
        elif self.encoded_path == "latent":
            Z = z_sample.unsqueeze(1).expand(batch_size, n_trgt, self.r_dim)
            dec_inp = Z
        elif self.encoded_path == "deterministic":
            dec_inp = R

        return dec_inp

    def decode(self, dec_inp, X_trgt):
        """
        Compute predicted distribution conditioned on representation and
        target positions.

        Parameters
        ----------
        dec_inp : torch.Tensor, size=[batch_size, n_trgt, inp_dim]
            Input to the decoder. `inp_dim` is `r_dim * 2 + x_dim` if
            `encoded_path == "both"` else `r_dim + x_dim`.
        """

        # size = [batch_size, n_trgt, x_transf_dim]
        X_transf = self.x_encoder(X_trgt)

        # size = [batch_size, n_trgt, y_dim*2]
        suff_stat_Y_trgt = self.decoder(X_transf, dec_inp)

        loc_trgt, scale_trgt = suff_stat_Y_trgt.split(self.y_dim, dim=-1)
        # Following convention "Empirical Evaluation of Neural Process Objectives"
        scale_trgt = self.min_std + (1 - self.min_std) * F.softplus(scale_trgt)
        p_y = Independent(self.PredictiveDistribution(loc_trgt, scale_trgt), 1)

        return p_y

    def set_extrapolation(self, min_max):
        """Set the neural process for extrapolation. Useful for  child classes."""
        pass


class AttentiveNeuralProcess(NeuralProcess):
    """
    Wrapper around `NeuralProcess` that implements an attentive neural process [4].

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    attention : callable or str, optional
        Type of attention to use. More details in `get_attender`.

    is_relative_pos : bool, optional
        Whether to add some relative position encodings. If `True` the positional
        encoding of size `kq_size` should be given in the `forward pass`. Only possible
        if `attention` takes `is_relative_pos` as argument
        (`{"multihead", "transformer"}`). Note that still not equivarian because
        use the position for the key and query.


    kwargs :
        Additional arguments to `NeuralProcess`.

    References
    ----------
    [4] Kim, Hyunjik, et al. "Attentive neural processes." arXiv preprint
        arXiv:1901.05761 (2019).
    """

    def __init__(self, x_dim, y_dim,
                 attention="scaledot",
                 encoded_path="both",
                 is_relative_pos=False,
                 **kwargs):

        super().__init__(x_dim, y_dim, encoded_path=encoded_path, **kwargs)

        self.attender = get_attender(attention, self.x_transf_dim, self.r_dim,
                                     self.r_dim, is_relative_pos=is_relative_pos)
        self.is_relative_pos = is_relative_pos
        if self.is_relative_pos:
            self.rel_pos_encoder = RelativeSinusoidalEncodings(x_dim, self.r_dim)

        self.reset_parameters()

    def deterministic_path(self, X_cntxt, Y_cntxt, X_trgt):
        """
        Deterministic encoding path.
        """
        # size = [batch_size, n_cntxt, x_transf_dim]
        keys = self.x_encoder(X_cntxt)

        # size = [batch_size, n_cntxt, r_dim]
        values = self.xy_encoder(keys, Y_cntxt)

        if X_trgt is None:
            r = self.aggregator(values, dim=1)
            # transforming, don't expand the representation : batch_size, r_dim
            return r

        # size = [batch_size, n_trgt, r_dim]
        queries = self.x_encoder(X_trgt)

        rel_pos_enc = self.rel_pos_encoder(X_cntxt, X_trgt) if self.is_relative_pos else None

        # size = [batch_size, n_trgt, value_size]
        R_attn = self.attender(keys, queries, values, rel_pos_enc=rel_pos_enc)

        return R_attn, None


class GlobalNeuralProcess(NeuralProcess):
    """
    Wrapper around `NeuralProcess` that implements a global neural process.
    I.e. with temporary queries to represent the functional space.

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    n_tmp_queries : int, optional
        Number of temporary queries to use.

    keys_to_tmp_attn : callable or str, optional
        Type of attention to use for {key} -> {tmp_query}. More details in
        `get_attender`.

    TmpSelfAttn : callable, optional
        Self attention mechanism to use for {tmp_query} -> {tmp_query}. Note
        that the temporary queries will be uniformly sampled and you can thus use
        a convolution instead. Example:
            - `partial(CNN, is_chan_last=True)` : uses a multilayer CNN. To
            be compatible with self attention the channel layer should be last.
            - `SelfAttention` : uses a self attention layer.

    keys_to_tmp_attn : callable or str, optional
        Type of attention to use. More details in `get_attender`.

    is_skip_tmp:
        Whether to have a residual connection that directly computes a mapping
        between keys and queries without temporary values (i.e. using
        classical cross attention). This will probably help optimisation.

    is_use_x : bool, optional
        Whether to encode and use X in the representation (r_i) and when decoding.
        If `False`, then guarantees translation equivariance (if add some
        representation of the positional differences) or invariance.

    is_encode_xy : bool, optional
        Whether to encode x and y.

    kwargs :
        Additional arguments to `NeuralProcess`.
    """

    def __init__(self, x_dim, y_dim,
                 n_tmp_queries=256,
                 keys_to_tmp_attn=SetConv,
                 TmpSelfAttn=partial(UnetCNN,
                                     Conv=nn.Conv1d,
                                     Pool=torch.nn.MaxPool1d,
                                     upsample_mode="linear",
                                     n_layers=10,
                                     is_double_conv=True,
                                     bottleneck=None,
                                     is_depth_separable=True,
                                     Normalization=nn.Identity,
                                     is_chan_last=True,
                                     kernel_size=7),
                 tmp_to_queries_attn=partial(SetConv, RadialBasisFunc=GaussianRBF),
                 is_skip_tmp=False,
                 is_use_x=False,
                 get_cntxt_trgt=CntxtTrgtGetter(is_add_cntxts_to_trgts=False),
                 is_encode_xy=False,
                 **kwargs):

        self.is_skip_tmp = is_skip_tmp
        super().__init__(x_dim, y_dim,
                         encoded_path="deterministic",
                         get_cntxt_trgt=get_cntxt_trgt,
                         is_use_x=is_use_x,
                         **kwargs)
        self.is_encode_xy = is_encode_xy
        if not self.is_encode_xy:
            self.xy_encoder = None
            value_size = y_dim
        else:
            value_size = self.r_dim

        self.n_tmp_queries = n_tmp_queries
        self.tmp_queries = torch.linspace(-1, 1, self.n_tmp_queries)
        self.keys_to_tmp_attender = get_attender(keys_to_tmp_attn, self.x_transf_dim,
                                                 value_size, self.r_dim)
        if TmpSelfAttn is not None:
            self.tmp_self_attn = TmpSelfAttn(self.r_dim)
            self.tmp_to_queries_attn = get_attender(tmp_to_queries_attn, self.x_transf_dim,
                                                    self.r_dim, self.r_dim)
        else:
            self.tmp_self_attn = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.is_skip_tmp:
            self.gate = torch.nn.Parameter(torch.tensor([0.] * self.r_dim))
            init_param_(self.gate)

    def deterministic_path(self, X_cntxt, Y_cntxt, X_trgt):
        """
        Deterministic encoding path.
        """
        # effectively puts on cuda only once
        self.tmp_queries = self.tmp_queries.to(X_cntxt.device)
        tmp_queries = self.tmp_queries.view(1, -1, 1)

        # size = [batch_size, n_cntxt, x_transf_dim]
        keys = self.x_encoder(X_cntxt)

        if self.is_encode_xy:
            # size = [batch_size, n_cntxt, r_dim]
            values = self.xy_encoder(keys, Y_cntxt)
        else:
            values = Y_cntxt

        # size = [batch_size, n_trgt, x_transf_dim]
        queries = self.x_encoder(X_trgt)

        if self.tmp_self_attn is None:
            # return a baseline without the tmp queries
            return self.keys_to_tmp_attender(keys, queries, values), None

        # size = [batch_size, n_tmp_trgt, r_dim]
        tmp_queries = tmp_queries.expand(X_cntxt.size(0), tmp_queries.size(1), self.x_transf_dim)

        if X_trgt is None:
            self.tmp_self_attn._is_summary = True

        # size = [batch_size, n_tmp, r_dim]
        tmp_values = self.keys_to_tmp_attender(keys, tmp_queries, values)  # , density
        tmp_values = torch.relu(tmp_values)
        tmp_values, summary = self.tmp_self_attn(tmp_values)  # , density)

        if X_trgt is None:
            return summary

        tmp_values = torch.relu(tmp_values)

        # size = [batch_size, n_trgt, r_dim]
        R_attn = self.tmp_to_queries_attn(tmp_queries, queries, tmp_values)  # _

        if self.is_skip_tmp:
            R_attn = R_attn + torch.sigmoid(self.gate) * self.keys_to_tmp_attender(keys, queries, values)

        return torch.relu(R_attn), summary

    def set_extrapolation(self, min_max):
        """
        Scale the temporary queries to be in a given range while keeping
        the same density than during training (used for extrapolation.).
        """
        self.tmp_queries = torch.linspace(-1, 1, self.n_tmp_queries)  # reset
        current_min = -1
        current_max = 1

        delta = self.tmp_queries[1] - self.tmp_queries[0]
        n_queries_per_increment = len(self.tmp_queries) / (current_max - current_min)
        n_add_left = math.ceil((current_min - min_max[0]) * n_queries_per_increment)
        n_add_right = math.ceil((min_max[1] - current_max) * n_queries_per_increment)

        tmp_queries_l = []
        if n_add_left > 0:
            tmp_queries_l.append(torch.arange(min_max[0], current_min, delta))
        tmp_queries_l.append(self.tmp_queries)

        if n_add_right > 0:
            # add delta to not have twice the previous max boundary
            tmp_queries_l.append(torch.arange(current_max, min_max[1], delta) + delta)
        self.tmp_queries = torch.cat(tmp_queries_l)
