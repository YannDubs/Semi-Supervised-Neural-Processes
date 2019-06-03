import torch
import torch.nn as nn
import torch.nn.functional as F

from skorch.predefined import MLP

from skssl.utils.torhextend import min_max_scale, MultivariateNormalDiag

d


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
    Implements Neural Process for functions of arbitrary dimensions.

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    r_dim : int
        Dimension of representation.

    Encoder: nn.Module
        Encoder module which maps {[x_i,y_i]} -> {r_i}. It should be constructed
        via `encoder(n_in, n_out)`.

    Decoder: nn.Module
        Decoder module which maps {r, x_t} -> {y_hat_t}. It should be constructed via
        `encoder(n_in, n_out)`.

    aggregator: callable
        Agregreator function which maps {r_i} -> r. It should have a an argument
        `dim` to say specify the dimensions of aggregation. The dimension should
        not be kept (i.e. keepdim=False).
    """

    def __init__(self, x_dim, y_dim,
                 Encoder=lambda *args: MLP(*args, hidden_size=128, n_hidden_layers=3),
                 Decoder=lambda *args: MLP(*args, hidden_size=128, n_hidden_layers=3),
                 r_dim=128,
                 aggregator=torch.mean):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim

        self.encoder = Encoder(x_dim + y_dim, r_dim)
        self.aggregator = aggregator
        self.decoder = Decoder(r_dim + x_dim, y_dim * 2)  # *2 because mean and var

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

        Y_cntxt: torch.Tensor, size=[batch_size, n_trgt, y_dim]
            Set of all target values {y_t}. Required during training.

        Return
        ------
        mu_trgt: torch.Tensor, size=[batch_size, n_trgt, y_dim]
            Mean predicted target.

        std_trgt: torch.Tensor, size=[batch_size, n_trgt, y_dim]
            Standard deviation predicted target.
        """
        batch_size, n_cntxt, _ = X_cntxt.shape()
        n_trgt = X_trgt.size(1)

        # batch_size, n_cntxt, r_dim
        R_cntxt = self.encoder(torch.cat((X_cntxt, Y_cntxt), dim=2))
        # batch_size, r_dim
        r = self.aggregator(R_cntxt, dim=1)
        # batch_size, n_trgt, r_dim (repeat no cupy)
        r = r.unsqueeze(1).expand(batch_size, n_trgt, self.r_dim)
        # batch_size, n_trgt, y_dim*2
        suff_stat_Y_hat_trgt = self.decoder(torch.cat((r, X_trgt), dim=-1))

        mu_trgt, std_trgt = suff_stat_Y_hat_trgt.split(self.y_dim, dim=-1)
        # Following convention "Empirical Evaluation of Neural Process Objectives"
        # and "Attentive Neural Processes"
        std_trgt = 0.1 + 0.9 * F.softplus(std_trgt)

        # for prediction you want mean tgt, for transform you want r
        # reirects Y_trgt to be used in the loss function
        return mu_trgt, r, MultivariateNormalDiag(mu_trgt, std_trgt), Y_trgt


class SpatialNeuralProcess(nn.Module):
    """
    Wraps regular Neural Process for Spatial representation.

    Parameters
    ----------
    x_shape: tuple of ints
        The first dimension represents the output, the rest are the spatial
        dimensions. E.g. (3, 32, 32) for images would mean output is dimensional
        (channels) while features are on a 2d grid.

    kwargs:
        Additional arguments to `NeuralProcess`.
    """

    def __init__(self, x_shape, Encoder, Decoder, r_dim=128):
        super().__init__()
        self.x_shape = x_shape
        self.spatial_dim = len(self.x_shape[1:])
        self.y_dim = self.x_shape[0]
        self.neural_process = NeuralProcess(self.spatial_dim, self.y_dim)

    def forward(self, X, context_mask, target_mask):
        """
        Given an image and masks of context and target points, returns a
        distribution over pixel intensities at the target points.

        Parameters
        ----------
        X: torch.Tensor, size=[*self.x_shape]
            Spatial input

        context_mask: torch.ByteTensor, size=[*x_shape[1:]]
            Binary mask indicating the context. Number of non zero should
            be same for all batch.

        target_mask: torch.ByteTensor, size=[*x_shape[1:]]
            Binary mask indicating the targets. Number of non zero should
            be same for all batch.
        """
        X_cntxt, Y_cntxt = self._apply_spatial_mask(X context_mask)
        X_trgt, Y_trgt = self._apply_spatial_mask(X, target_mask)
        return self.neural_process(X_cntxt, Y_cntxt, X_trgt, Y_trgt=Y_trgt)

    def _apply_spatial_mask(self, x, mask):
        batch_size, *spatial_shape = mask.shape

        # batch_size, self.spatial_dim
        nonzero_idx = mask.nonzero()
        # assume same amount of nonzero acros batch
        n_cntxt = mask[0].nonzero().size(0)

        # don't take Y because same for all
        x_masked = nonzero_idx[:, 1:].view(batch_size, n_cntxt, self.spatial_dim).float()
        # normalize to [-1,1] every dimension
        x_masked = min_max_scale(x_masked, min_val=-1, max_val=1, dim=0)

        mask = mask.unsqueeze(1).expand(batch_size, self.y_dim, *spatial_shape)
        y_masked = x[mask].view(batch_size, self.y_dim, n_cntxt)
        # batch_size, n_cntxt, self.y_dim
        y_masked = y_masked.permute(0, 2, 1)

        return x_masked, y_masked
