import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from skssl.predefined import MLP
from skssl.utils.torchextend import min_max_scale, MultivariateNormalDiag
from skssl.utils.helpers import prod, ratio_to_int
from skssl.training.masks import random_masker, no_masker


__all__ = ["NeuralProcessLoss", "SpatialNeuralProcess"]


class NeuralProcessLoss(nn.Module):
    """

    """

    def forward(self, inputs, y=None, X=None):
        """Compute the Neural Process Loss.

        Parameters
        ----------
        inputs : tuple
            Tuple of (mu_trgt, r, pred_dist, *). This can directly take the
            output of NueralProcess.

        y: None
            Placeholder.

        X: None
            Placeholder.
        """

        pred_dist = inputs[2]
        Y_trgt = inputs[3]

        X = X["X"] if isinstance(X, dict) else X
        assert X is not None
        reconstruct, z_sample, z_suff_stat = inputs[:3]
        rec_loss = reconstruction_loss(X, reconstruct, distribution=self.distribution)
        kl_loss = kl_normal_loss(z_suff_stat)
        loss = rec_loss + self.beta * kl_loss

        if weight is not None:
            loss = loss * weight

        log_p = dist.log_prob(target_y)

        return loss.mean(dim=0)


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

    References
    ----------
    [1] Garnelo, Marta, et al. "Neural processes." arXiv preprint arXiv:1807.01622 (2018).
    [2] Garnelo, Marta, et al. "Conditional neural processes." arXiv preprint
        arXiv:1807.01613 (2018).
    [3] Le, Tuan Anh, et al. "Empirical Evaluation of Neural Process Objectives."
        NeurIPS workshop on Bayesian Deep Learning. 2018.
    """

    def __init__(self, x_dim, y_dim,
                 Encoder=lambda *args: MLP(*args, hidden_size=128, n_hidden_layers=3),
                 Decoder=lambda *args: MLP(*args, hidden_size=128, n_hidden_layers=3),
                 r_dim=128,
                 aggregator=torch.mean,
                 LatentEncoder=None):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim

        self.encoder = Encoder(x_dim + y_dim, r_dim)
        self.aggregator = aggregator
        self.decoder = Decoder(r_dim + x_dim, y_dim * 2)  # *2 because mean and var

        if LatentEncoder is not None:
            self.lat_encoder = LatentEncoder(r_dim, r_dim * 2)
        else:
            self.lat_encoder = None

        self.reset_parameters()

    def forward(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        """
        Given context pairs (x_i, y_i) and target points x_t, return the
        sufficient statistics of distribution over target points y_trgt.

        Parameters
        ----------
        X_cntxt: torch.Tensor, size=[batch_size, n_cntxt, x_dim]
            Set of all context features {x_i} . Note that x_cntxt is a
            subset of x_trgt.

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

        mean_trgt: torch.Tensor, size=[batch_size, n_trgt, y_dim]
            Mean predicted target.

        std_trgt: torch.Tensor, size=[batch_size, n_trgt, y_dim]
            Standard deviation predicted target.

        Y_trgt: torch.Tensor, size=[batch_size, n_trgt, y_dim]
            Set of all target values {y_t}, returned to redirect it to the loss
            function.

        mean_z: torch.Tensor, size=[batch_size, r_dim]
            Mean latent.

        std_z: torch.Tensor, size=[batch_size, r_dim]
            Standard deviation of latent.
        """
        batch_size, n_cntxt, _ = X_cntxt.shape()
        n_trgt = X_trgt.size(1)

        # batch_size, n_cntxt, r_dim
        R_cntxt = self.encoder(torch.cat((X_cntxt, Y_cntxt), dim=-1))
        # batch_size, r_dim
        r = self.aggregator(R_cntxt, dim=1)
        z_sample, mean_z, std_z = self.encode_latent(r)
        # batch_size, n_trgt, r_dim (repeat no copy)
        z_expanded = z_sample.unsqueeze(1).expand(batch_size, n_trgt, self.r_dim)
        # batch_size, n_trgt, y_dim*2
        suff_stat_Y_hat_trgt = self.decoder(torch.cat((z_expanded, X_trgt), dim=-1))
        mean_trgt, std_trgt = suff_stat_Y_hat_trgt.split(self.y_dim, dim=-1)
        # Following convention "Empirical Evaluation of Neural Process Objectives"
        # and "Attentive Neural Processes"
        std_trgt = 0.1 + 0.9 * F.softplus(std_trgt)

        # for transform you want r (could also want mu_z but r should have this info)
        return r, mean_trgt, std_trgt, Y_trgt, mean_z, std_z

    def reset_parameters(self):
        weights_init(self)

    def encode_latent(self, r):
        """Converts the representation to a latent distribution and sample from it."""
        if self.lat_encoder is not None:
            z_suff_stat = self.lat_encoder(X)
            # Define sigma following convention in "Empirical Evaluation of Neural
            # Process Objectives". SO doesn't use usual logvar
            #mean_z, std_z = suff_stat.view(z_suff_stat.shape[0], -1, 2).unbind(-1)
            #std = 0.1 + 0.9 * torch.sigmoid(std)
            #z_sample = reparameterize_meanstd(mean, std, is_sample=self.training)
        else:
            z_sample = r
            mean_z = None
            std_z = None

        return z_sample, mean_z, std_z


class SpatialNeuralProcess(nn.Module):
    """
    Wraps regular Neural Process for Spatial representation.

    Parameters
    ----------
    x_shape: tuple of ints
        The first dimension represents the number of outputs, the rest are the
        spatial dimensions. E.g. (3, 32, 32) for images would mean output is 3
        dimensional (channels) while features are on a 2d grid.

    context_masker: callable, optional
        Get the context masks if not given during the forward call.

    context_masker: callable, optional
        Get the context masks if not given during the forward call.

    kwargs:
        Additional arguments to `NeuralProcess`.
    """

    def __init__(self, x_shape,
                 context_masker=random_masker,
                 target_masker=no_masker,
                 **kwargs):
        super().__init__()
        self.x_shape = x_shape
        self.spatial_dim = len(self.x_shape[1:])
        self.y_dim = self.x_shape[0]
        self.neural_process = NeuralProcess(self.spatial_dim, self.y_dim, **kwargs)
        self.context_masker = context_masker
        self.target_masker = target_masker

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X, context_mask=None, target_mask=None):
        """
        Given an image and masks of context and target points, returns a
        distribution over pixel intensities at the target points.

        Parameters
        ----------
        X: torch.Tensor, size=[batch_size, *self.x_shape]
            Spatial input

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

        X_cntxt, Y_cntxt = self._apply_spatial_mask(X, context_mask)
        X_trgt, Y_trgt = self._apply_spatial_mask(X, target_mask)
        return self.neural_process(X_cntxt, Y_cntxt, X_trgt, Y_trgt=Y_trgt)

    def _apply_spatial_mask(self, X, mask):
        batch_size, *spatial_shape = mask.shape

        # batch_size, self.spatial_dim
        nonzero_idcs = mask.nonzero()
        # assume same amount of nonzero across batch
        n_cntxt = mask[0].nonzero().size(0)

        # first dim is batch idx.
        X_masked = nonzero_idcs[:, 1:].view(batch_size, n_cntxt, self.spatial_dim).float()
        # normalize spatial idcs to [-1,1]
        for i, size in enumerate(spatial_shape):
            X_masked[:, :, i] *= 2 / (size - 1)  # in [0,2]
            X_masked[:, :, i] -= 1  # in [-1,1]

        mask = mask.unsqueeze(1).expand(batch_size, self.y_dim, *spatial_shape)
        Y_masked = X[mask].view(batch_size, self.y_dim, n_cntxt)
        # batch_size, n_cntxt, self.y_dim
        Y_masked = Y_masked.permute(0, 2, 1)

        return X_masked, Y_masked
