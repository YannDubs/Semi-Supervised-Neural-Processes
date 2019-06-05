"""
Framework for semi supervised learning VAE with auxilary variable from:

Maaløe, Lars, et al. "Auxiliary deep generative models."
arXiv preprint arXiv:1602.05473 (2016).
"""
import torch

from skssl.predefined import MLP, WideResNet, ReversedSimpleCNN
from skssl.predefined.helpers import add_flat_input
from skssl.utils.initialization import weights_init
from skssl.utils.helpers import HyperparameterInterpolator
from skssl.training.helpers import split_labelled_unlabelled
from .sslvae import SSLVAELoss, SSLVAE


class SSLAuxVAELoss(SSLVAELoss):
    """
    Compute the SSL VAE loss as in [1].

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence. If 1, corresponds to standard VAE. By default
        linear annealing from 0 to 1 in 200 steps.

    distribution : {"bernoulli", "gaussian", "laplace"}, optional
        Distribution of the likelihood for each feature. See `reconstruction_loss`
        for more details.

    kwargs:
        Additional arguments to `VAELoss.

    References
    ----------
        [1] Kingma, Durk P., et al. "Semi-supervised learning with deep generative
        models." Advances in neural information processing systems. 2014.
    """

    def __init__(self, beta=HyperparameterInterpolator(0, 1, 200), **kwargs):
        super().__init__(beta=beta, **kwargs)
        self.criterion_y = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    def forward(self, inputs, y=None, X=None):
        """Compute the SSL VAE loss.

        Note
        ----
        - DOesn't compute the loss due to p(y) because doesn't depend on param

        Parameters
        ----------
        inputs : tuple
            Tuple of (y_hat, z_sample, reconstruct, *). This can directly take the output
            of VAE.

        y : None
            Placeholder.

        X : torch.Tensor, size = [batch_size, *x_shape]
            Training data corresponding to the targets.
        """
        assert X is not None
        assert y is not None
        # reverts back to have (y_hat, reconstruct, z_sample, *) like for VAELoss
        inputs = inputs[:1] + inputs[2:0:-1] + inputs[2:]

        X_lab, X_unlab = split_labelled_unlabelled(X, y)
        # the first n_lab are the labelled ones and the rest are unlabelled
        inputs_lab, inputs_unlab = split_labelled_unlabelled(inputs, y, is_ordered=True)
        labelled_loss = unlabelled_loss = 0

        if (y != -1).sum() != 0:
            labelled_loss = self._labelled_loss(inputs_lab[1:], X_lab)
        if (y == -1).sum() != 0:
            unlabelled_loss = self._unlabelled_loss(inputs_unlab[1:], X_unlab, inputs_lab[0])

        # - log q(y|x)
        classifier_loss = self.criterion_y(inputs[0], y)

        return labelled_loss + unlabelled_loss + self.alpha * classifier_loss

    def _unlabelled_loss(self, vae_inputs, X, y_hat):
        """for unlabelled data: E_y [labelled loss] - H[q(y|x)]"""
        p_y_hat = F.softmax(y_hat, dim=-1)
        n_unlab, y_dim = y_hat.shape

        def get_vae_inputs(i):
            # unlabelled vae inputs are batch concatenated for all labels
            return tuple(inp[i * n_unlab:(i + 1) * n_unlab] for inp in vae_inputs)

        expected_lab_loss = sum(self._labelled_loss(get_vae_inputs(i), X, weight=p_y_hat[:, i])
                                for i in range(y_dim))
        # H[q(y|x)] = -dot(q,log(q))
        ent_y = - torch.bmm(p_y_hat.view(-1, 1, y_dim),
                            F.log_softmax(y_hat, dim=-1).view(-1, y_dim, 1)
                            ).mean()
        return expected_lab_loss - ent_y

    def _labelled_loss(self, vae_inputs, X, **kwargs):
        """for labelled data: - log p(x|y,z) + beta KL(q(z|x,y)||p(z))"""
        return super().forward(vae_inputs, X=X, **kwargs)


# MODEL
class SSLAuxVAE(SSLVAE):
    """Semisupervised VAE with auxilary variabl from [1] (M2 model).
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
        Number of auxilary dimensions.

    AuxEncoder: nn.Module, optional
        Encoder module which maps x -> a. It should be callable with
        `aux_encoder(x_shape, n_out)`.

    Encoder: nn.Module, optional
        Encoder module which maps x,a,y -> z. It should be callable with
        `encoder(x_shape, a_dim+y_dim, n_out)`. If you have an encoder that
        maps x -> z you can convert it via `add_flat_input(Encoder)`.

    Decoder: nn.Module, optional
        Decoder module which maps z,y -> x. It should be callable with
        `decoder(z_dim+y_dim, x_shape)`. No non linearities should be applied to the
        output.

    AuxDecoder: nn.Module, optional
        Decoder module which maps x,z,y -> a. It should be callable with
        `aux_decoder(x_shape, z_dim+y_dim, a_dim)`. If you have an Decoder that
        maps x -> a you can convert it via `add_flat_input(Decoder)`.

    Classifier: nn.Module, optional
        Classifier module which maps x,a -> y. It should be callable with
        `encoder(x_shape, a_dim, n_out)`. If you have an encoder that
        maps x -> y you can convert it via `add_flat_input(Encoder)`.

    Reference
    ---------
    [1] Maaløe, Lars, et al. "Auxiliary deep generative models."
        arXiv preprint arXiv:1602.05473 (2016).
    """

    def __init__(self, x_shape, y_dim,
                 z_dim=64, a_dim=64,
                 AuxEncoder=WideResNet,
                 Encoder=add_flat_input(WideResNet),
                 Decoder=ReversedSimpleCNN,
                 AuxDecoder=add_flat_input(WideResNet),
                 Classifier=add_flat_input(WideResNet)):

        super().__init__()
        self.x_shape = x_shape
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.a_dim = a_dim

        self.aux_encoder = AuxEncoder(x_shape, a_dim * 2)
        self.encoder = Encoder(x_shape, a_dim + y_dim, z_dim)
        self.decoder = Decoder(z_dim + y_dim, x_shape)
        self.aux_decoder = AuxDecoder(z_dim + y_dim, x_shape)
        self.classifier = Classifier(x_shape, a_dim, y_dim)

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

        y : torch.Tensor, size = [batch_size]
            Labels. -1 for unlabelled. `None` if all unlabelled.

        Returns
        ------
        y_hat: torch.Tensor, size = [batch_size, y_dim]
            Multinomial logits (i.e. no softmax).

        z_sample: torch.Tensor, size = [batch_size, z_dim]
            Latent sample.

        reconstruct: torch.Tensor, size = [batch_size, *x_shape]
            Reconstructed image. Values between 0,1 (i.e. after logistic).

        z_suff_stat: torch.Tensor, size = [batch_size, z_dim*2]
            Sufficient statistics of the latent sample {mu; logvar}.
        """
        if y is None:
            y = torch.tensor([-1], device=X.device).expand(X.size(0))

        X_lab, X_unlab = split_labelled_unlabelled(X, y)
        y_lab, y_unlab = split_labelled_unlabelled(y, y)

        # q(a|x)
        q_a_suff_stat = self.aux_decoder(X)
        q_a_sample = self.reparameterize(q_a_suff_stat)

        # q(y|a,x)
        y_hat = self.classifier(X, q_a_sample)

        # initialize as empty for concatenation
        empty = torch.tensor([], dtype=y_hat.dtype, device=y_hat.device)
        rec_lab = z_suff_stat_lab = z_sample_lab = empty
        rec_unlab = z_suff_stat_unlab = z_sample_unlab = empty

        if (y != -1).sum() != 0:
            y_lab_onehot = torch.zeros_like(y_hat[:y_lab.size(0), ...]
                                            ).scatter_(1, y_lab.view(-1, 1), 1)
            rec_lab, z_sample_lab, z_suff_stat_lab, = self._forward_labelled(X_lab, y_lab_onehot)

        if (y == -1).sum() != 0:
            # no copying
            y_unlab_onehot = torch.zeros_like(y_hat[:1, ...]).expand(y_unlab.size(0), self.y_dim)
            rec_unlab, z_sample_unlab, z_suff_stat_unlab, = self._forward_unlabelled(X_unlab, y_unlab_onehot)

        reconstruct = torch.cat((rec_lab, rec_unlab), dim=0)
        z_sample = torch.cat((z_suff_stat_lab, z_suff_stat_unlab), dim=0)
        z_suff_stat = torch.cat((z_suff_stat_lab, z_suff_stat_unlab), dim=0)

        # inverts z_suff_stat, reconstruct because second output (z_sample) is
        # used for .transform while first output (y_hat) is used for .predict
        return y_hat, z_sample, reconstruct, z_suff_stat

    def _forward_labelled(self, X, q_a_sample, y_onehot):
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
        return reconstruct, z_sample, z_suff_stat, p_a_sample, p_a_suff_stat

    def _forward_unlabelled(self, X, y_onehot):
        """Same as for labelled but marginalizes out labels => compute outputs for all y"""
        reconstruct_list, z_sample_list, z_suff_stat_list = [], [], []

        # expectation over all possible labels
        for l in range(self.y_dim):
            y_onehot.zero_()
            y_onehot[:, l] = 1

            reconstruct, z_sample, z_suff_stat = self._forward_labelled(X, y_onehot)
            reconstruct_list.append(reconstruct)
            z_suff_stat_list.append(z_suff_stat)
            z_sample_list.append(z_sample)

        reconstruct = torch.cat(reconstruct_list, dim=0)
        z_suff_stat = torch.cat(z_suff_stat_list, dim=0)
        z_sample = torch.cat(z_sample_list, dim=0)

        return reconstruct, z_sample, z_suff_stat

    def sample_decode(self, z, y):
        """
        Returns a sample from the decoder.

        Parameters
        ----------
        z : torch.Tensor, size = [batch_size, z_dim]
            Latent variable.

        y : torch.Tensor, size = [batch_size]
            Labels.
        """
        batch_size = z.size(0)
        y_onehot = torch.zeros(batch_size, self.y_dim, device=z.device).float()
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        sample = torch.sigmoid(self.decoder(torch.cat((z, y_onehot), dim=1)))
        return sample
