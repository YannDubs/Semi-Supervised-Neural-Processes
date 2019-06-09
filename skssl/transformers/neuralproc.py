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


__all__ = ["NeuralProcessLoss", "GridNeuralProcess", "NeuralProcess"]


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
            Tuple of (r, p_y_trgt, Y_trgt, q_z_trgt, q_z_cntxt). This can directly take the
            output of NueralProcess.

        y: None
            Placeholder.

        weight: torch.Tensor, size = [batch_size,]
            Weight of every example. If None, every example is weighted by 1.
        """
        r, p_y_trgt, Y_trgt, q_z_trgt, q_z_cntxt = inputs
        batch_size = Y_trgt.size(0)

        neg_log_like = - p_y_trgt.log_prob(Y_trgt).view(batch_size, -1).sum(-1)

        if q_z_cntxt is not None:
            # use latent variables
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
    functions of arbitrary  dimensions.

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    r_dim : int, optional
        Dimension of representation.

    Encoder: nn.Module, optional
        Encoder module which maps {[x_i;y_i]} -> {r_i}. It should be constructed
        via `encoder(n_in, n_out)`.

    Decoder: nn.Module, optional
        Decoder module which maps {[r;x_t]} -> {y_hat_t}. It should be constructed via
        `decoder(n_in, n_out)`.

    aggregator: callable, optional
        Agregreator function which maps {r_i} -> r. It should have a an argument
        `dim` to say specify the dimensions of aggregation. The dimension should
        not be kept (i.e. keepdim=False).

    LatentEncoder: nn.Module, optional
        Encoder which maps r -> z_suff_stat. It should be constructed via
        `LatentEncoder(r_dim, n_out)`. If not given, the model will be a Conditional
        Neural Process [2] (no latents).

    get_cntxt_trgt: callable, optional
        Function that split the input into context and target points.
        `X_cntxt, Y_cntxt, X_trgt, Y_trgt = self.get_cntxt_trgt(X, **kwargs)


    References
    ----------
        [1] Garnelo, Marta, et al. "Neural processes." arXiv preprint
        arXiv:1807.01622 (2018).
        [2] Garnelo, Marta, et al. "Conditional neural processes." arXiv preprint
            arXiv:1807.01613 (2018).
        [3] Le, Tuan Anh, et al. "Empirical Evaluation of Neural Process Objectives."
            NeurIPS workshop on Bayesian Deep Learning. 2018.
    """

    def __init__(self, x_dim, y_dim,
                 Encoder=_DeepMLP,
                 Decoder=_DeepMLP,
                 r_dim=128,
                 aggregator=torch.mean,
                 LatentEncoder=None,
                 get_cntxt_trgt=context_target_split):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim

        self.encoder = Encoder(self.x_dim + self.y_dim, self.r_dim)
        self.aggregator = aggregator
        self.decoder = Decoder(self.r_dim + self.x_dim, self.y_dim * 2)  # *2 because mean and var

        if LatentEncoder is not None:
            self.lat_encoder = LatentEncoder(self.r_dim, self.r_dim * 2)
        else:
            self.lat_encoder = None

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
            Set of all target features {x_t}.

        Y_trgt: torch.Tensor, size=[batch_size, n_trgt, y_dim], optional
            Set of all target values {y_t}. Only required during training.

        Return
        ------
        r: torch.Tensor, size=[batch_size, r_dim]
            Representation.

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
        r, z_sample, q_z_cntxt = self.encode(X_cntxt, Y_cntxt)
        p_y_trgt = self.decode(z_sample, X_trgt)

        if self.training and self.lat_encoder is not None:
            _, _, q_z_trgt = self.encode(X_trgt, Y_trgt)
        else:
            q_z_cntxt, q_z_trgt = None, None

        # for transform you want r (could also want mean_z but r should have this info)
        return r, p_y_trgt, Y_trgt, q_z_trgt, q_z_cntxt

    def encode(self, X, Y):
        # batch_size, n_cntxt, r_dim
        R_cntxt = self.encoder(torch.cat((X, Y), dim=-1))
        # batch_size, r_dim
        r = self.aggregator(R_cntxt, dim=1)
        z_sample, q_z = self._encode_latent(r)
        return r, z_sample, q_z

    def decode(self, z, X_trgt):
        """
        Given latents and target positions, return the predicted distribution.

        Parameters
        ----------
        z: torch.Tensor, size=[batch_size, r_dim]
            Latent samples

        X_trgt: torch.Tensor, size=[batch_size, n_trgt, x_dim]
            Set of all target features {x_t}.
        """
        batch_size, n_trgt, _ = X_trgt.shape

        Z = z.unsqueeze(1).expand(batch_size, n_trgt, self.r_dim)
        # batch_size, n_trgt, y_dim*2
        suff_stat_Y_trgt = self.decoder(torch.cat((Z, X_trgt), dim=-1))
        mean_trgt, std_trgt = suff_stat_Y_trgt.split(self.y_dim, dim=-1)
        # Following convention "Empirical Evaluation of Neural Process Objectives"
        std_trgt = 0.1 + 0.9 * F.softplus(std_trgt)
        p_y = MultivariateNormalDiag(mean_trgt, std_trgt)
        return p_y

    def _encode_latent(self, r):
        """Converts the representation to a latent distribution and sample from it."""
        if self.lat_encoder is not None:
            z_suff_stat = self.lat_encoder(r)
            # Define sigma following convention in "Empirical Evaluation of Neural
            # Process Objectives". SO doesn't use usual logvar
            mean_z, std_z = z_suff_stat.view(z_suff_stat.shape[0], -1, 2).unbind(-1)
            std_z = 0.1 + 0.9 * torch.sigmoid(std_z)
            z_dist = MultivariateNormalDiag(mean_z, std_z)
            # sample even when not training
            z_sample = z_dist.rsample()
        else:
            z_sample = r
            z_dist = None

        return z_sample, z_dist


class GridNeuralProcess(NeuralProcess):
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
        Get the context masks if not given during the forward call.

    is_union_trgt_cntxt: bool, optional
        Whether to use all cntxt points as targets. This has been shown emperically
        to give better results.

    kwargs:
        Additional arguments to `NeuralProcess`.
    """

    def __init__(self, x_shape,
                 context_masker=random_masker,
                 target_masker=no_masker,
                 is_union_trgt_cntxt=True,
                 **kwargs):
        super().__init__(len(x_shape[1:]), x_shape[0],
                         get_cntxt_trgt=None,  # redefines it
                         **kwargs)
        self.x_shape = x_shape
        self.context_masker = context_masker
        self.target_masker = target_masker
        self.is_union_trgt_cntxt = is_union_trgt_cntxt

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
            `self.target_masker(batch_size, mask_shape)`.
        """
        batch_size = X.size(0)
        device = X.device

        if context_mask is None:
            context_mask = self.context_masker(batch_size, *self.x_shape[1:]).to(device)
        if target_mask is None:
            target_mask = self.target_masker(batch_size, *self.x_shape[1:]).to(device)

        if self.is_union_trgt_cntxt:
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
