import ast
import argparse

from torch.optim import Adam
import torch.nn as nn
from sklearn.pipeline import Pipeline

from skorch.callbacks import ProgressBar, Checkpoint, TrainEndCheckpoint

from skssl.training import NeuralNetEstimator, NeuralNetClassifier, get_supervised_iterator
from skssl.transformers.vae import VAE, VAELoss
from skssl.predefined import (WideResNet, ReversedWideResNet, ReversedSimpleCNN,
                              SimpleCNN, MLP)
from skssl.classifiers.sslvae import SSLVAELoss, SSLVAE
from skssl.utils.helpers import FixRandomSeed
from utils.data.datasets import get_dataset, get_train_dev_test_ssl
from utils.helpers import FormatterNoDuplicate


def load_pretrained(model, dataset, **kwargs):
    """Load a pretrained model.

    Parameters
    ----------
    model: {"vae"}
        Type of mdoel to load.

    dataset: str
        Name of the dataset to load. Argument to `get_train_dev_test_ssl`.

    kwargs:
        Additional arguments to `pretrained_*`,`instantiate_mod_chckpt`,
        `NeuralNetEstimator`.
    """
    if model == "vae":
        return _pretrained_vae(dataset, **kwargs)
    if model == "supervised":
        return _pretrained_supervised(dataset, **kwargs)
    else:
        raise ValueError("Unkown model={}.".format(model))


def _get_enc_dec(enc_dec):
    """Return the correct encoder and decoder."""
    if enc_dec == "resnet":
        encoder = WideResNet
        decoder = ReversedWideResNet
    elif enc_dec == "resnetEnc_cnnDec":
        encoder = WideResNet
        decoder = ReversedSimpleCNN
    elif enc_dec == "cnn":
        encoder = SimpleCNN
        decoder = ReversedSimpleCNN
    elif enc_dec == "mlp":
        encoder = MLP
        decoder = MLP
    else:
        raise ValueError("Unkown enc_dec={}.".format(enc_dec))

    return encoder, decoder


def add_vae(net, is_vae, dataset, z_dim):
    if is_vae:
        vae = load_pretrained("vae", dataset, z_dim=z_dim).freeze()
        return Pipeline([('m1', vae), ('classifier', net)])
    else:
        return net


def _pretrained_supervised(dataset,
                           is_vae=False,
                           is_retrain=False,
                           max_epochs=100,
                           z_dim=64,
                           **kwargs):
    """Load a pretrained vae.

    Parameters
    ----------
    z_dim: int, optional
        Number of latent dimensions.

    is_vae: bool, optional
        Whether to predict on the output of a vae instead of directly on the image.

    is_retrain: bool, optional
        Whether to retrain the model and save it, instead of loading a pretraine one.

    max_epochs: int, optional
        Number of epochs if `is_retrain`.

    z_dim: int, optional
        Number of latent dimensions. Only used if `is_vae`.
    """

    if is_retrain:
        train, dev, test = get_train_dev_test_ssl(dataset)
        x_shape = train.shape
        n_classes = train.n_classes
    else:
        dev = None
        x_shape = get_dataset(dataset).shape
        n_classes = get_dataset(dataset).n_classes

    model = MLP(z_dim, n_classes, hidden_size=z_dim) if is_vae else WideResNet(x_shape, n_classes)

    net, chckpt = instantiate_mod_chckpt(model,
                                         nn.CrossEntropyLoss,
                                         "supervised/isvae{}_{}".format(is_vae, dataset),
                                         Trainer=NeuralNetClassifier,
                                         is_ssl=False,
                                         dev=dev,
                                         max_epochs=max_epochs,
                                         iterator_train=get_supervised_iterator,
                                         **kwargs)
    if is_retrain:
        net = add_vae(net, is_vae, dataset, z_dim)
        net.fit(train, train.targets)
    else:
        net.initialize()
        net.load_params(checkpoint=chckpt)
        net = add_vae(net, is_vae, dataset, z_dim)

    return net


def _pretrained_vae(dataset,
                    z_dim=64,
                    enc_dec="resnetEnc_cnnDec",
                    is_retrain=False,
                    max_epochs=100,
                    **kwargs):
    """Load a pretrained vae.

    Parameters
    ----------
    z_dim: int, optional
        Number of latent dimensions.

    enc_dec: {"resnet","cnn","resnetEnc_cnnDec"}, optional
        Type of encoder and decoder. Models are defined in `skssl.predefined`

    is_retrain: bool, optional
        Whether to retrain the model and save it, instead of loading a pretraine one.

    max_epochs: int, optional
        Number of epochs if `is_retrain`.
    """

    if is_retrain:
        train, dev, test = get_train_dev_test_ssl(dataset)
        x_shape = train.shape
    else:
        dev = None
        x_shape = get_dataset(dataset).shape

    encoder, decoder = _get_enc_dec(enc_dec)
    vae = VAE(encoder, decoder, x_shape, z_dim=z_dim)

    net, chckpt = instantiate_mod_chckpt(vae, VAELoss, "vae/{}_{}".format(enc_dec, dataset),
                                         is_ssl=False,
                                         dev=dev,
                                         criterion__distribution="laplace",
                                         max_epochs=max_epochs,
                                         **kwargs)
    if is_retrain:
        net.fit(train)
    else:
        net.initialize()
        net.load_params(checkpoint=chckpt)

    return net


def _pretrained_sslvae(dataset,
                       mode="m2",
                       is_retrain=False,
                       max_epochs=100,
                       z_dim=64,
                       **kwargs):
    """Load a pretrained vae.

    Parameters
    ----------
    z_dim: int, optional
        Number of latent dimensions.

    mode: {"m2","m1+m2"}, optional
        Type of semi supervised vae. M1+M2 first uses a vae then the semi supervised
        vae, while M2 uses directly the semi supervised vae.

    enc_dec: {"resnet","cnn","resnetEnc_cnnDec"}, optional
        Type of encoder and decoder. Models are defined in `skssl.predefined`

    is_retrain: bool, optional
        Whether to retrain the model and save it, instead of loading a pretraine one.

    max_epochs: int, optional
        Number of epochs if `is_retrain`.
    """

    if is_retrain:
        train, dev, test = get_train_dev_test_ssl(dataset)
        x_shape = train.shape
    else:
        dev = None
        x_shape = get_dataset(dataset).shape

    if mode == "m2":
        encoder, decoder = _get_enc_dec("resnetEnc_cnnDec")
    elif mode == "m1+m2":
        DeepMLP = lambda *args: MLP(*args, n_hidden_layers=3)
        encoder, decoder = DeepMLP, DeepMLP
        x_shape = (z_dim, )
        z_dim = z_dim // 2
    else:
        raise ValueError()

    sslvae = SSLVAE(make_ssl_encoder(encoder), decoder, x_shape, z_dim=z_dim)

    net, chckpt = instantiate_mod_chckpt(sslvae, SSLVAE, "sslvae/{}_{}".format(mode, dataset),
                                         is_ssl=True,
                                         dev=dev,
                                         max_epochs=max_epochs,
                                         **kwargs)

    if is_retrain:
        net = add_vae(net, mode == "m1+m2", dataset, z_dim)
        net.fit(train, train.targets)
    else:
        net.initialize()
        net.load_params(checkpoint=chckpt)
        net = add_vae(net, mode == "m1+m2", dataset, z_dim)

    return net


def instantiate_mod_chckpt(model, criterion, chckpt_name,
                           Trainer=NeuralNetEstimator,
                           dev=None,
                           basedir="results/pretrained/",
                           seed=123,
                           **kwargs):
    """Instantiaite the model and checkpoint.

    Parameters
    ----------
    model: nn.Module

    criterion: nn.Module

    chckpt_name: str
        Name of the checkpoint file.

    dev: torch.utils.data.Dataset, optional
        Dev dataset.

    basedir: str, optional
        Base directory where the checpoint will be saved.

    seed: int, optional
        Random seed for deterministic training.

    kwargs:
        Additional arguments to NeuralNetEstimator
    """
    chckpt = Checkpoint(dirname=basedir + chckpt_name)
    net = Trainer(model,
                  criterion=criterion,
                  devset=dev,
                  iterator_train__shuffle=True,
                  callbacks=[ProgressBar(),
                             chckpt,
                             TrainEndCheckpoint(dirname=basedir + chckpt_name,
                                                fn_prefix='train_end_'),
                             FixRandomSeed(seed)],
                  **kwargs)

    return net, chckpt


if __name__ == '__main__':
    description = "CLI for pretraining a model"
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    parser.add_argument('model', type=str, help="Type of model to train.")
    parser.add_argument("dataset", type=str,
                        help="Name of the dataset to load. Argument to `get_train_dev_test_ssl`.")
    parser.add_argument('-k', '--kwargs', type=ast.literal_eval, default={},
                        help='Additional arguments to `pretrained_*`,`instantiate_mod_chckpt`, `NeuralNetEstimator`. Should be written as a python dict "{"key1": "value1"}"')

    args = parser.parse_args()
    load_pretrained(args.model, args.dataset, is_retrain=True, **args.kwargs)
