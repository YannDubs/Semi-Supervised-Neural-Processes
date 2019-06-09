import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence

from skssl.predefined import MLP
from skssl.utils.initialization import weights_init
from skssl.utils.torchextend import min_max_scale, MultivariateNormalDiag
from skssl.training.masks import random_masker, no_masker, or_masks
from skssl.training.helpers import context_target_split


__all__ = ["NeuralProcessLoss", "make_grid_neural_process", "NeuralProcess",
           "AttentiveNeuralProcess"]


# LOSS
class NeuralProcessLoss(nn.Module):
    """
    Compute the Neural Process Loss [1].

    Parameters
    ----------
    get_beta : callable, optional
        Function which returns the weight of the kl divergence given `is_training`
        . Returning a constant 1 corresponds to standard VAE.

    References
    ----------
    [1] Garnelo, Marta, et al. "Neural processes." arXiv preprint
        arXiv:1807.01622 (2018).
    """

    def __init__(self, get_beta=lambda _: 1):
        super().__init__()
        self.get_beta = get_beta

    def forward(self, inputs, y=None, weight=None):
        """Compute the Neural Process Loss averaged over the batch.

        Parameters
        ----------
        inputs: tuple
            Tuple of (p_y_trgt, Y_trgt, q_z_trgt, q_z_cntxt). This can directly take the
            output of NueralProcess.

        y: None
            Placeholder.

        weight: torch.Tensor, size = [batch_size,]
            Weight of every example. If None, every example is weighted by 1.
        """
        p_y_trgt, Y_trgt, q_z_trgt, q_z_cntxt = inputs
        batch_size = Y_trgt.size(0)

        neg_log_like = - p_y_trgt.log_prob(Y_trgt).view(batch_size, -1).sum(-1)

        if q_z_trgt is not None:
            # use latent variables and training
            # note that during validation the kl will actually be 0 because
            # we do not compute q_z_trgt
            kl_loss = kl_divergence(q_z_trgt, q_z_cntxt).view(batch_size, -1).sum(-1)
        else:
            kl_loss = 0

        loss = neg_log_like + self.get_beta(self.training) * kl_loss

        if weight is not None:
            loss = loss * weight

        return loss.mean(dim=0)


def _DeepMLP(*args):
    return MLP(*args, hidden_size=128, n_hidden_layers=3)


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

    Encoder : nn.Module, optional
        Encoder module which maps {[x_i;y_i]} -> {r_i}. It should be constructed
        via `encoder(n_in, n_out)`.

    Decoder : nn.Module, optional
        Decoder module which maps {[r;x_t]} -> {y_hat_t}. It should be constructed via
        `decoder(n_in, n_out)`.

    aggregator : callable, optional
        Agregreator function which maps {r_i} -> r. It should have a an argument
        `dim` to say specify the dimensions of aggregation. The dimension should
        not be kept (i.e. keepdim=False).

    LatentEncoder : nn.Module, optional
        Encoder which maps r -> z_suff_stat. It should be constructed via
        `LatentEncoder(r_dim, n_out)`. Only used if `encoded_path in ["latent",
        "both"]`.

    get_cntxt_trgt : callable, optional
        Function that split the input into context and target points.
        `X_cntxt, Y_cntxt, X_trgt, Y_trgt = self.get_cntxt_trgt(X, **kwargs).
        Note: context points should be a subset of target ones.

    encoded_path : {"deterministic", "latent", "both"}
        Which path(s) to use. If `"deterministic"` uses a Conditional Neural
        Process [2] (no latents), where the decoder gets a deterministic representation
        as input (function of the context). If `"latent"` uses the original a
        Neural Process [1], where the decoder gets a sample latent representation
        as input (function of the target during training and context during test).
        If `"both"` concatenates both representations as described in [4].

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
    """

    def __init__(self, x_dim, y_dim,
                 Encoder=_DeepMLP,
                 Decoder=_DeepMLP,
                 r_dim=128,
                 aggregator=torch.mean,
                 LatentEncoder=MLP,
                 get_cntxt_trgt=context_target_split,
                 encoded_path="deterministic"):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.encoded_path = encoded_path.lower()
        self.is_transform = False

        rz_dim = self.r_dim * 2 if self.encoded_path == "both" else self.r_dim

        self.encoder = Encoder(self.x_dim + self.y_dim, self.r_dim)
        self.aggregator = aggregator
        self.decoder = Decoder(rz_dim + self.x_dim, self.y_dim * 2)  # *2 because mean and var

        if self.encoded_path in ["latent", "both"]:
            self.lat_encoder = LatentEncoder(self.r_dim, self.r_dim * 2)
        elif self.encoded_path == "deterministic":
            self.lat_encoder = None
        else:
            raise ValueError("Unkown encoded_path={}.".format(encoded_path))

        if get_cntxt_trgt is not None:
            self.get_cntxt_trgt = get_cntxt_trgt

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X, y=None, **kwargs):
        X_cntxt, Y_cntxt, X_trgt, Y_trgt = self.get_cntxt_trgt(X, y, **kwargs)
        return self.forward_step(X_cntxt, Y_cntxt, X_trgt, Y_trgt=Y_trgt)

    def forward_step(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        """
        Given context pairs (x_i, y_i) and target points x_t, return the
        sufficient statistics of distribution over target points y_trgt.

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
        representation : torch.Tensor, size=[batch_size, r_dim]
            Only returned when `is_transform` and not training (and returns only
            this).

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
            r_lat, z_sample, q_z_cntxt = self.latent_path(X_cntxt, Y_cntxt)

            if self.training:
                # during training when we know Y_trgt, we compute the latent using
                # he targets as context. If we used it for the deterministic path,
                # then the model would cheat by learning a point representation for
                # each function => bad representation
                _, z_sample, q_z_trgt = self.latent_path(X_trgt, Y_trgt)

        if self.is_transform and not self.training:
            if self.encoded_path in ["latent", "both"]:
                representation = r_lat
            if self.encoded_path == "deterministic":
                # representation for tranform should be indep of target
                representation = deterministic_path(X_cntxt, Y_cntxt, None)
            # for transform you want representation (could also want mean_z but r
            # should have this info).
            return representation

        if self.encoded_path in ["deterministic", "both"]:
            R_det = self.deterministic_path(X_cntxt, Y_cntxt, X_trgt)

        dec_inp = self.make_dec_inp(R_det, z_sample, X_trgt)
        p_y_trgt = self.decode(dec_inp)

        return p_y_trgt, Y_trgt, q_z_trgt, q_z_cntxt

    def latent_path(self, X, Y):
        """Latent encoding path."""
        # batch_size, n_cntxt, r_dim
        R_cntxt = self.encoder(torch.cat((X, Y), dim=-1))
        # batch_size, r_dim
        r = self.aggregator(R_cntxt, dim=1)

        z_suff_stat = self.lat_encoder(r)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives". SO doesn't use usual logvar
        mean_z, std_z = z_suff_stat.view(z_suff_stat.shape[0], -1, 2).unbind(-1)
        std_z = 0.1 + 0.9 * torch.sigmoid(std_z)
        q_z = MultivariateNormalDiag(mean_z, std_z)
        # sample even when not training
        z_sample = q_z.rsample()

        return r, z_sample, q_z

    def deterministic_path(self, X_cntxt, Y_cntxt, X_trgt):
        """
        Deterministic encoding path. `X_trgt` can be used in child classes
        to give a target specific representation (e.g. attentive neural processes.
        """
        # batch_size, n_cntxt, r_dim
        R_cntxt = self.encoder(torch.cat((X_cntxt, Y_cntxt), dim=-1))
        # batch_size, r_dim
        r = self.aggregator(R_cntxt, dim=1)

        if X_trgt is None:
            return r

        batch_size, n_trgt, _ = X_trgt.shape
        R = r.unsqueeze(1).expand(batch_size, n_trgt, self.r_dim)
        return R

    def make_dec_inp(self, R, z_sample, X_trgt):
        """
        Make the input for the decoder. deterministic: [R;X], latent: [Z;X],
        both: [R;Z;X].
        """
        batch_size, n_trgt, _ = X_trgt.shape

        dec_inp = X_trgt

        if self.encoded_path in ["both", "latent"]:
            Z = z_sample.unsqueeze(1).expand(batch_size, n_trgt, self.r_dim)
            dec_inp = torch.cat((Z, dec_inp), dim=-1)

        if self.encoded_path in ["both", "deterministic"]:
            dec_inp = torch.cat((R, dec_inp), dim=-1)

        return dec_inp

    def decode(self, dec_inp):
        """
        Compute predicted distribution conditioned on representation and
        target positions.

        Parameters
        ----------
        dec_inp : torch.Tensor, size=[batch_size, n_trgt, inp_dim]
            Input to the decoder. `inp_dim` is `r_dim * 2 + x_dim` if
            `encoded_path == "both"` else `r_dim + x_dim`.
        """
        # batch_size, n_trgt, y_dim*2
        suff_stat_Y_trgt = self.decoder(dec_inp)
        mean_trgt, std_trgt = suff_stat_Y_trgt.split(self.y_dim, dim=-1)
        # Following convention "Empirical Evaluation of Neural Process Objectives"
        std_trgt = 0.1 + 0.9 * F.softplus(std_trgt)
        p_y = MultivariateNormalDiag(mean_trgt, std_trgt)
        return p_y


class AttentiveNeuralProcess(NeuralProcess):
    """
    Wrapper around `NeuralProcess` that implements an attentive neural process [4].

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    kwargs :
        Additional arguments to `NeuralProcess`.

    References
    ----------
    [4] Kim, Hyunjik, et al. "Attentive neural processes." arXiv preprint
        arXiv:1901.05761 (2019).
    """

    def __init__(self, x_dim, y_dim, **kwargs):
        pass

    def deterministic_path(self, X_cntxt, Y_cntxt, X_trgt):
        """
        Deterministic encoding path. `X_trgt` can be used in child classes
        to give a target specific representation (e.g. attentive neural processes.
        """
        # batch_size, n_cntxt, r_dim
        R_cntxt = self.encoder(torch.cat((X_cntxt, Y_cntxt), dim=-1))
        # batch_size, r_dim
        r = self.aggregator(R_cntxt, dim=1)

        if X_trgt is None:
            return r

        batch_size, n_trgt, _ = X_trgt.shape
        R = r.unsqueeze(1).expand(batch_size, n_trgt, self.r_dim)
        return R


def make_grid_neural_process(NP):
    class GridNeuralProcess(NP):
        """
        Neural Process for Grid inputs.

        Parameters
        ----------
        x_shape: tuple of ints
            The first dimension represents the number of outputs, the rest are the
            grid dimensions. E.g. (3, 32, 32) for images would mean output is 3
            dimensional (channels) while features are on a 2d grid.

        context_masker: callable, optional
            Get the context masks if not given during the forward call.

        target_masker: callable, optional
            Get the context masks if not given during the forward call. Note that the
            context points will always be added to targets.

        kwargs:
            Additional arguments to `NeuralProcess`.
        """

        def __init__(self, x_shape,
                     context_masker=random_masker,
                     target_masker=no_masker,
                     **kwargs):
            super().__init__(len(x_shape[1:]), x_shape[0],
                             get_cntxt_trgt=None,  # redefines it
                             **kwargs)
            self.x_shape = x_shape
            self.context_masker = context_masker
            self.target_masker = target_masker

        def get_cntxt_trgt(self, X, y=None, context_mask=None, target_mask=None):
            """
            Given an image and masks of context and target points, returns a
            distribution over pixel intensities at the target points.

            Parameters
            ----------
            X: torch.Tensor, size=[batch_size, *self.x_shape]
                Grid input

            y: None
                Placeholder

            context_mask: torch.ByteTensor, size=[batch_size, *x_shape[1:]]
                Binary mask indicating the context. Number of non zero should
                be same for all batch. If `None` generates it using
                `self.context_masker(batch_size, mask_shape)`.

            target_mask: torch.ByteTensor, size=[batch_size, *x_shape[1:]]
                Binary mask indicating the targets. Number of non zero should
                be same for all batch. If `None` generates it using
                `self.target_masker(batch_size, mask_shape)`. Note that the
                context points will always be added to targets.
            """
            batch_size = X.size(0)
            device = X.device

            if context_mask is None:
                context_mask = self.context_masker(batch_size, *self.x_shape[1:]).to(device)
            if target_mask is None:
                target_mask = self.target_masker(batch_size, *self.x_shape[1:]).to(device)

            # add context points to targets: has been shown emperically better
            target_mask = or_masks(target_mask, context_mask)

            X_cntxt, Y_cntxt = self._apply_grid_mask(X, context_mask)
            X_trgt, Y_trgt = self._apply_grid_mask(X, target_mask)
            return X_cntxt, Y_cntxt, X_trgt, Y_trgt

        def _apply_grid_mask(self, X, mask):
            batch_size, *grid_shape = mask.shape

            # batch_size, x_dim
            nonzero_idcs = mask.nonzero()
            # assume same amount of nonzero across batch
            n_cntxt = mask[0].nonzero().size(0)

            # first dim is batch idx.
            X_masked = nonzero_idcs[:, 1:].view(batch_size, n_cntxt, self.x_dim).float()
            # normalize grid idcs to [-1,1]
            for i, size in enumerate(grid_shape):
                X_masked[:, :, i] *= 2 / (size - 1)  # in [0,2]
                X_masked[:, :, i] -= 1  # in [-1,1]

            mask = mask.unsqueeze(1).expand(batch_size, self.y_dim, *grid_shape)
            Y_masked = X[mask].view(batch_size, self.y_dim, n_cntxt)
            # batch_size, n_cntxt, self.y_dim
            Y_masked = Y_masked.permute(0, 2, 1)

            return X_masked, Y_masked

    return GridNeuralProcess
