"""
Framework for semi supervised learning VAE with auxiliary variable from:

Maaløe, Lars, et al. "Auxiliary deep generative models."
arXiv preprint arXiv:1602.05473 (2016).
"""
import torch

from skssl.predefined import MLP, WideResNet, ReversedWideResNet
from skssl.predefined.helpers import add_flat_input
from skssl.utils.initialization import weights_init
from skssl.utils.torchextend import kl_normal_loss
from skssl.utils.helpers import HyperparameterInterpolator
from skssl.training.helpers import split_labelled_unlabelled
from .sslvae import SSLVAELoss, SSLVAE

__all__ = ['SSLAuxVAELoss', 'SSLAuxVAE']


class SSLAuxVAELoss(SSLVAELoss):
    """
    Compute the SSL VAE with Auxilary variable loss as in [1].

    Parameters
    ----------
    get_beta: callable, optional
        Function which returns the weight of the kl divergence given `is_training`
        . By default linear annealing from 0 to 1 in 200 steps.

    kwargs:
        Additional arguments to `SSLVAELoss`.

    References
    ----------
        [1] Maaløe, Lars, et al. "Auxiliary deep generative models."
        arXiv preprint arXiv:1602.05473 (2016).
    """

    def __init__(self, get_beta=HyperparameterInterpolator(0, 1, 200), **kwargs):
        super().__init__(get_beta=get_beta, **kwargs)

    def _split_inputs(self, inputs):
        """Split the inputs."""
        pred_logits, inputs_lab, inputs_unlab = inputs[0], inputs[1:6], inputs[6:11]
        add_out = inputs[11] if len(inputs) > 11 else {}
        return pred_logits, inputs_lab, inputs_unlab, add_out

    def _labelled_loss(self, inputs, X,
                       weight=None,
                       q_a_sample=None,
                       q_a_suff_stat=None,
                       **kwargs):
        """for labelled data: - log p(x|y,z) + beta KL(q(z|x,y)||p(z)) + KL(q(a|x)||p(a|x,y,z))"""
        assert q_a_sample is not None and q_a_suff_stat is not None
        (*vae_inputs), p_a_sample, p_a_suff_stat = inputs

        ssl_vae_loss = super()._labelled_loss(vae_inputs, X, weight=weight)

        auxiliary_kl = kl_normal_loss(q_a_suff_stat, p_a_suff_stat)
        if weight is not None:
            auxiliary_kl = auxiliary_kl * weight
        auxiliary_kl = auxiliary_kl.mean(dim=0)

        return ssl_vae_loss + auxiliary_kl

# MODEL


class SSLAuxVAE(SSLVAE):
    """Semisupervised VAE with auxiliary variabl from [1] (M2 model).
    This is a classifier, so the predicted label is the output of
    `NeuralNet.predict`. Using `NeuralNet.transform` will return the latent
    representation. `SSLVAE.sample_decode` can still be used for generation.

    Parameters
    ----------
    x_shape: array-like
        Shape of a single example x.

    y_dim: int
        Number of classes.

    z_dim: int, optional
        Number of latent dimensions.

    a_dim: int, optional
        Number of auxiliary dimensions.

    AuxEncoder: nn.Module, optional
        Encoder module which maps X(_transf) -> a_suff_stat. It should be callable
        with `aux_encoder(x_shape, n_out)`.

    Encoder: nn.Module, optional
        Encoder module which maps X(_transf), [a;y] -> z_suff_stat. It should be
        callable with `encoder(x_shape, a_dim+y_dim, n_out)`. If you have an
        encoder that maps x -> z you can convert it via `add_flat_input(Encoder)`.

    Decoder: nn.Module, optional
        Decoder module which maps [z;y] -> X. It should be callable with
        `decoder(z_dim+y_dim, x_shape)`. No non linearities should be applied to the
        output.

    AuxDecoder: nn.Module, optional
        Decoder module which maps X(_transf), [z;y] -> _suff_stat. It should be
        callable with `aux_decoder(x_shape, z_dim+y_dim, a_dim)`. If you have an
        Decoder that maps x -> a you can convert it via `add_flat_input(Decoder)`.

    Classifier: nn.Module, optional
        Classifier module which maps X(_transf), a -> y. It should be callable with
        `encoder(x_shape, a_dim, n_out)`. If you have an encoder that
        maps x -> y you can convert it via `add_flat_input(Encoder)`.

    kwargs:
        Additional arguments to `SSLVAE` (e.g. Transformer).

    Reference
    ---------
    [1] Maaløe, Lars, et al. "Auxiliary deep generative models."
        arXiv preprint arXiv:1602.05473 (2016).
    """

    def __init__(self, x_shape, y_dim,
                 z_dim=64,
                 a_dim=64,
                 AuxEncoder=MLP,
                 Encoder=add_flat_input(MLP),
                 Decoder=ReversedWideResNet,
                 AuxDecoder=add_flat_input(MLP),
                 Classifier=add_flat_input(MLP),
                 **kwargs):

        empty = lambda *args: None
        super().__init__(x_shape, y_dim, z_dim=z_dim, Encoder=empty,
                         Decoder=empty, Classifier=empty, **kwargs)

        transform_dim = self.transform_dim if self.transform_dim is not None else x_shape
        self.a_dim = a_dim
        self.aux_encoder = AuxEncoder(transform_dim, self.a_dim * 2)
        self.encoder = Encoder(transform_dim, self.a_dim + self.y_dim, self.z_dim * 2)
        self.decoder = Decoder(self.z_dim + self.y_dim, self.x_shape)
        self.aux_decoder = AuxDecoder(transform_dim, self.z_dim + self.y_dim, self.a_dim * 2)
        self.classifier = Classifier(transform_dim, self.a_dim, self.y_dim)

        self.reset_parameters()

    def classify(self, X, q_a_sample=None, **kwargs):
        """Classify the given X by returning multinomial logits."""
        # q(y|a,x)
        pred_logits = self.classifier(X, q_a_sample)
        return pred_logits

    def _get_additional_outputs(self, X):
        """Function that can be overriden to add outputs to the forward method of SSLVAE."""
        # q(a|x)
        q_a_suff_stat = self.aux_encoder(X)
        q_a_sample = self.reparameterize(q_a_suff_stat)
        return {"q_a_sample": q_a_sample, "q_a_suff_stat": q_a_suff_stat}

    def _forward_labelled(self, X, y_onehot, q_a_sample=None, **kwargs):
        """Forward fucntion for labbeled examples."""
        # q(z|a,y,x)
        ay = torch.cat([q_a_sample, y_onehot], dim=1)
        z_suff_stat = self.encoder(X, ay)
        z_sample = self.reparameterize(z_suff_stat)
        # p(x|z,y)
        zy = torch.cat((z_sample, y_onehot), dim=1)
        reconstruct = torch.sigmoid(self.decoder(zy))
        # p(a|z,y,x)
        p_a_suff_stat = self.aux_decoder(X, zy)
        p_a_sample = self.reparameterize(p_a_suff_stat)
        return z_sample, z_suff_stat, reconstruct, p_a_sample, p_a_suff_stat

    def forward(self, X, y=None):
        """
        Forward pass of model.

        Parameters
        ----------
        X : torch.Tensor, size = [batch_size, *x_shape]
            Batch of data.

        y : torch.Tensor, size = [batch_size]
            Labels. -1 for unlabelled. `None` if all unlabelled.

        Returns
        -------
        pred_logits: torch.Tensor, size = [batch_size, y_dim]
            Multinomial logits (i.e. no softmax).

        z_sample_lab: torch.Tensor, size = [n_lab, z_dim]
            Latent sample of labelled data.

        z_suff_stat_lab: torch.Tensor, size = [n_lab, z_dim*2]
            Sufficient statistics of the labelled latent sample {mu; logvar}.

        reconstruct_lab: torch.Tensor, size = [n_lab, *x_shape]
            Reconstructed labelled image. Values between 0,1 (i.e. after logistic).

        p_a_sample_lab: torch.Tensor, size = [n_lab, a_dim]
            Sample of the generative auxiliary variable for labelled data.

        p_a_suff_stat_lab: torch.Tensor, size = [n_lab, a_dim]
           Sufficient statistics of the labelled generative auxiliary variable {mu; logvar}.

        z_sample_unlab: torch.Tensor, size = [n_unlab, y_dim, z_dim]
            Latent sample of labelled data. Stacked on first dimension for
            each possible y (in order).

        z_suff_stat_unlab: torch.Tensor, size = [n_unlab, y_dim, z_dim*2]
            Sufficient statistics of the labelled latent sample {mu; logvar}.
            Stacked on first dimension for each possible y (in order).

        reconstruct_unlab: torch.Tensor, size = [n_unlab, y_dim, *x_shape]
            Reconstructed labelled image. Values between 0,1 (i.e. after logistic).
            Stacked on first dimension for each possible y (in order).

        p_a_sample_unlab: torch.Tensor, size = [n_unlab, a_dim]
            Sample of the generative auxiliary variable for unlabelled data.

        p_a_suff_stat_unlab: torch.Tensor, size = [n_unlab, a_dim]
           Sufficient statistics of the unlabelled generative auxiliary variable {mu; logvar}.
        """
        # changes the doc
        return super().forward(X, y=y)
