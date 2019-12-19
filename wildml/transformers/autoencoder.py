import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import RelaxedBernoulli, Bernoulli, Independent, Normal
from torch.distributions.kl import kl_divergence

from essl.predefined import get_Cnn
from essl.utils.initialization import weights_init
from essl.utils.distributions import MultivariateNormalDiag, straight_through, NoDistribution
from essl.utils.encdec import SimpleCNN, ReversedSimpleCNN

__all__ = ["AutoencoderLoss", "Autoencoder"]


# LOSS
class AutoencoderLoss(nn.Module):
    """
    Compute an Beta-(V)AE loss.
    
    Parameters
    ----------
    get_beta : callable, optional
        Function which returns the weight of the kl divergence given `is_training`
        . Returning a constant 1 corresponds to standard VAE.
    
    distribution : {"bernoulli", "gaussian", "laplace"}, optional
        Distribution of the likelihood for each feature. See `reconstruction_loss`
        for more details.
    """

    def __init__(self, get_beta=lambda _: 1, is_return_all=False, distribution="gaussian"):
        super().__init__()
        self.get_beta = get_beta
        self.is_return_all = is_return_all
        self.distribution = distribution

    def forward(self, pred_outputs, Y_trgt, weight=None):
        """Compute the VAE loss averaged over the batch.

        Parameters
        ----------
        pred_outputs : tuple
            Tuple of (reconstruct, z_suff_stat, *). Output of VAE.

        Y_trgt: torch.Tensor, size=[batch_size, *x_shape]
            Set of all target values {y_t}.

        weight : torch.Tensor, size = [batch_size,]
            Weight of every example. If None, every example is weighted by 1.
        """
        reconstruction, q_z, *_ = pred_outputs
        batch_size, *x_shape = Y_trgt.shape

        if len(reconstruction) or reconstruction is None:
            rec_loss = 0
        else:
            if self.distribution == "bernoulli":
                rec_loss = F.binary_cross_entropy(reconstruction, Y_trgt, reduction="none")
            elif self.distribution == "gaussian":
                rec_loss = F.mse_loss(reconstruction, Y_trgt, reduction="none")
            else:
                raise ValueError("Unkown distribution={}.".format(distribution))

            rec_loss = rec_loss.view(batch_size, -1).sum(-1)

        if isinstance(q_z, Independent) and isinstance(q_z.base_dist, Normal):
            # gaussian q
            mean_0 = torch.zeros_like(q_z.base_dist.loc)
            std_1 = torch.ones_like(q_z.base_dist.scale)
            kl_loss = kl_divergence(q_z, MultivariateNormalDiag(mean_0, std_1))
        elif isinstance(q_z, Independent) and isinstance(q_z.base_dist, RelaxedBernoulli):
            unif = torch.ones_like(q_z.base_dist.probs) / 2
            kl_loss = kl_divergence(
                Independent(Bernoulli(q_z.base_dist.probs), 1), Independent(Bernoulli(unif), 1)
            )
        elif isinstance(q_z, NoDistribution) or q_z is None:
            kl_loss = torch.tensor([0.0], device=Y_trgt.device)
        else:
            raise ValueError(f"Unkown distribution form : {q_z}.")

        loss = rec_loss + self.get_beta(self.training) * kl_loss

        if weight is not None:
            loss = loss * weight

        if self.is_return_all:
            return loss

        return loss.mean(dim=0)


# MODEL


class Autoencoder(nn.Module):
    """Unsupervised (V)AE. This is a transformer as it is primarily
    used for representation learning, so the mean latent is the output of `predict`
    but `sample_decode` can still be used for generation.

    Parameters
    ----------
    x_shape : tuple of ints
        Shape of a single example x.

    Encoder : nn.Module, optional
        Encoder module which maps x -> z_suff_stat. It should be callable with
        `encoder(x_shape, n_out)`.

    Decoder : nn.Module, optional
        Decoder module which maps z -> x. It should be callable with
        `decoder(z_dim, x_shape)`. No non linearities should be applied to the
        output.

    q_distribution : {"gaussian", "bernoulli", None}, optional
        Form of the variational distribution.

    z_dim : int, optional
        Number of latent dimensions.
    """

    def __init__(
        self,
        x_shape,
        Encoder=SimpleCNN,
        Decoder=ReversedSimpleCNN,
        q_distribution="gaussian",
        z_dim=64,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.x_shape = x_shape
        self.q_distribution = q_distribution

        n_suff_stat = self.z_dim * 2 if self.q_distribution == "gaussian" else self.z_dim
        self.encoder = Encoder(x_shape, n_suff_stat)
        self.decoder = Decoder(self.z_dim, x_shape)
        self.is_transform = False

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X, y=None):
        """
        Forward pass of model.
        
        Parameters
        ----------
        X : torch.Tensor, size = [batch_size, *x_shape]
            Batch of data.
        
        y: None
            Placeholder
        
        Returns
        ------
        reconstruct: torch.Tensor, size = [batch_size, *x_shape]
            Reconstructed image. Values between 0,1 (i.e. after logistic).
        
        q_z: torch.Tensor, size = [batch_size, z_dim]
            Latent distribution.
        """
        z_suff_stat = self.encoder(X)

        if self.q_distribution == "gaussian":
            mean_z, logvar = z_suff_stat.view(z_suff_stat.shape[0], -1, 2).unbind(-1)
            std_z = torch.exp(0.5 * logvar)
            q_z = MultivariateNormalDiag(mean_z, std_z)
            z_sample = q_z.rsample()
            mode_z = mean_z
        elif self.q_distribution == "bernoulli":
            q_z = Independent(RelaxedBernoulli(temperature=0.67, logits=z_suff_stat), 1)
            z_sample = straight_through(q_z.rsample(), torch.round)
            mode_z = straight_through(q_z.base_dist.probs, torch.round)
        elif self.q_distribution == None:
            q_z = NoDistribution(z_suff_stat)
            mode_z = z_sample = z_suff_stat
        else:
            raise ValueError(f"Unkown q_distribution={self.q_distribution}.")

        # dirty trick because of skorch that needs tensor as output
        q_z.to = lambda *args, **kwargs: q_z

        if self.is_transform and not self.training:
            return mode_z

        reconstruct = torch.sigmoid(self.decoder(z_sample))

        return reconstruct, q_z, mode_z
