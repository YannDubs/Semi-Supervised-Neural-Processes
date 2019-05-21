from sklearn.base import ClassifierMixin

from skssl.predefined.vae import VAE, VAELoss
from skssl.utils.initialization import weights_init


class M2(VAE, ClassifierMixin):
    """Semisupervised VAE from [1] (M2 model). This is aclassifier, so the predicted
    label is the output of `predict` but `sample_decode` can still be used for generation.

    Parameters
    ----------
    encoder : nn.Module
        Encoder module which maps x,y -> z.

    decoder : nn.Module
        Decoder module which maps z,y -> x.

    classifier : nn.Module
        Classifier module which mapy X -> y. The last layer should be a softmax.

    x_shape : array-like
        Shape of a single example x.

    y_dim : int
        Number of classes.

    z_dim : int, optional
        Number of latent dimensions.

    Reference
    ---------
    [1] Kingma, D. P., Mohamed, S., Rezende, D. J., & Welling, M. (2014).
        Semi-supervised learning with deep generative models. In Advances in neural
        information processing systems (pp. 3581-3589).
    """

    def __init__(self, encoder, decoder, classifier, x_shape, y_dim, z_dim=10):
        super().__init__(None, None, x_shape, z_dim)
        # initialize here as need y_dim
        self.encoder = encoder(x_shape, y_dim, z_dim)
        self.decoder = decoder(x_shape, y_dim, z_dim)
        self.classifier = classifier(x_shape, y_dim)
        self.y_dim = y_dim

        self.reset_parameters()

    def forward(self, X, y):
        """
        Forward pass of model.

        Parameters
        ----------
        X : torch.Tensor, size = [*x_shape]
            Batch of data.

        y : torch.Tensor, size = [batch_size]
            Labels. -1 for unlabelled.
        """
        y_hat = self.classifier(X)
        z_dist = self.encoder(X, y)  # return [batch_size, n_classes] ???
        z_sample = self.reparameterize(*z_dist)
        reconstruct = self.decoder(z_sample, y)
        return y_hat, reconstruct, z_dist, z_sample

    def sample_decode(self, z, y):
        """
        Returns a sample from the decoder.

        Parameters
        ----------
        z : torch.Tensor, size = [batch_size, latent_dim]
            Latent variable.

        y : torch.Tensor, size = [batch_size, y_dim]
            One hot encoded labels.
        """
        y = y.float()
        sample = self.decoder(z, y)
        return sample
