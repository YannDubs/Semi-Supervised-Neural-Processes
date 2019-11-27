import os
import ast
import argparse

from torch.optim import Adam
import torch.nn as nn
from sklearn.pipeline import Pipeline

import skorch
from skorch.callbacks import ProgressBar, Checkpoint, TrainEndCheckpoint, EarlyStopping

from econvcnp.training import NeuralNetEstimator, NeuralNetClassifier, NeuralNetTransformer
from econvcnp.training.loaders import get_supervised_iterator
from econvcnp.transformers import VAE, VAELoss
from econvcnp.predefined import (WideResNet, ReversedWideResNet, ReversedSimpleCNN,
                              SimpleCNN, MLP, merge_flat_input)
from econvcnp.classifiers import SSLVAELoss, SSLVAE, SSLAuxVAELoss, SSLAuxVAE
from econvcnp.training.helpers import FixRandomSeed
from utils.data.ssldata import get_dataset, get_train_dev_test_ssl
from utils.helpers import FormatterNoDuplicate


def load_pretrained(model, dataset, **kwargs):
    """Load a pretrained model.

    Parameters
    ----------
    model: {"vae", "supervised", "sslvae"}
        Type of mdoel to load.

    dataset: str
        Name of the dataset to load. Argument to `get_train_dev_test_ssl`.

    kwargs:
        Additional arguments to `pretrained_*`,`instantiate_mod_chckpt`,
        `NeuralNetEstimator`.
    """
    if model == "vae":
        return _pretrained_vae(dataset, **kwargs)
    elif model == "supervised":
        return _pretrained_supervised(dataset, **kwargs)
    elif model == "sslvae":
        return _pretrained_sslvae(dataset, **kwargs)
    else:
        raise ValueError("Unkown model={}.".format(model))


# TRANSFORMERS
def _pretrained_vae(dataset,
                    transform_dim=64,
                    enc_dec="resnet",
                    **kwargs):
    """
    Load a pretrained vae.

    Parameters
    ----------
    dataset: str
        Name of dataset.

    transform_dim: int, optional
        Number of latent dimensions.

    enc_dec: {"mlp","resnet","cnn","resnetEnc_cnnDec"}, optional
        Type of encoder and decoder. Models are defined in `econvcnp.predefined`

    kwargs:
        Additional argumens to `_predefined_base`
    """
    chckpt_name = "vae/{}_z{}_{}".format(enc_dec, transform_dim, dataset)
    x_shape = get_dataset(dataset).shape

    Encoder, Decoder = _get_enc_dec(enc_dec)
    vae = VAE(x_shape, Encoder=Encoder, Decoder=Decoder, z_dim=transform_dim)

    model = _pretrained_base(vae, VAELoss, dataset, chckpt_name,
                             Trainer=NeuralNetTransformer,
                             **kwargs)

    return model


# CLASSIFIERS
def _pretrained_supervised(dataset,
                           transformer=None,
                           transform_dim=64,
                           **kwargs):
    """
    Load a pretrained supervised model.

    Parameters
    ----------
    dataset: str
        Name of dataset.

    tranformer: {None, "vae"}, optional
        Pretrained transformer to apply before the loaded supervised model.

    transform_dim: int, optional
        Number of dimensions after the transformer.

    kwargs:
        Additional argumens to `_predefined_base`
    """
    chckpt_name = "supervised/transf{}_{}".format(transformer, dataset)

    x_shape = get_dataset(dataset).shape
    n_classes = get_dataset(dataset).n_classes

    if transformer is None:
        model = WideResNet(x_shape, n_classes)
    else:
        model = MLP(transform_dim, n_classes, hidden_size=transform_dim)
        transformer = load_pretrained(transformer, dataset,
                                      transform_dim=transform_dim).freeze()

    model = _pretrained_base(model, nn.CrossEntropyLoss, dataset, chckpt_name,
                             transformer=transformer,
                             Trainer=NeuralNetClassifier,
                             is_ssl=False,
                             iterator_train=get_supervised_iterator,
                             **kwargs)

    return model


def _pretrained_sslvae(dataset,
                       z_dim=64,
                       mode="m2",
                       transf_dec="resnet",
                       **kwargs):
    """
    Load a pretrained semi supervised VAE.

    Parameters
    ----------
    dataset: str
        Name of dataset.

    z_dim: int, optional
        Number of latent dimensions.

    mode: {"m2","m1+m2", "auxiliary"}, optional
        Type of semi supervised vae.`"m1+m2"` first uses a vae then the semi supervised
        vae, while `"m2"` uses directly the semi supervised vae. `"auxiliary"` uses
        an auxiliary variable to be more expressive.

    transf_dec: {"mlp","resnet","cnn","resnetEnc_cnnDec"}, optional
        Type of transformer and decoder. Models are defined in
        `econvcnp.predefined`.

    kwargs:
        Additional argumens to `_predefined_base`
    """
    chckpt_name = "sslvae/{}_z{}_{}".format(mode, z_dim, dataset)

    x_shape = get_dataset(dataset).shape
    n_classes = get_dataset(dataset).n_classes

    mode = mode.lower()

    if mode == "auxiliary":
        Transformer, Decoder = _get_enc_dec(transf_dec)
        sslvae = SSLAuxVAE(x_shape, n_classes,
                           z_dim=z_dim,
                           a_dim=z_dim,
                           transform_dim=z_dim,
                           AuxEncoder=MLP,
                           Encoder=merge_flat_input(MLP),
                           Decoder=Decoder,
                           AuxDecoder=merge_flat_input(MLP),
                           Classifier=merge_flat_input(MLP),
                           Transformer=Transformer)
        Loss = SSLAuxVAELoss
        pipeline_transformer = None

    else:
        if mode == "m2":
            Transformer, Decoder = _get_enc_dec(transf_dec)
            Encoder = MLP
            transform_dim = z_dim
            pipeline_transformer = None
        elif mode == "m1+m2":
            Transformer, Encoder, Decoder = MLP, MLP, _DeepMLP
            transform_dim, x_shape = z_dim, z_dim
            pipeline_transformer = load_pretrained("vae", dataset,
                                                   transform_dim=z_dim).freeze()
        else:
            raise ValueError()

        sslvae = SSLVAE(x_shape, n_classes,
                        z_dim=z_dim,
                        transform_dim=transform_dim,
                        Encoder=merge_flat_input(Encoder),
                        Decoder=Decoder,
                        Classifier=Encoder,
                        Transformer=Transformer)
        Loss = SSLVAELoss

    model = _pretrained_base(sslvae, Loss, dataset, chckpt_name,
                             transformer=pipeline_transformer,
                             Trainer=NeuralNetClassifier,
                             **kwargs)

    return model


# HELPERS
def _pretrained_base(model, criterion, dataset, chckpt_name,
                     is_retrain=False,
                     transformer=None,
                     Trainer=NeuralNetEstimator,
                     basedir="results/pretrained/",
                     seed=123,
                     max_epochs=100,
                     is_all_labels=False,
                     patience=None,
                     suffix="",
                     **kwargs):
    """Instantiaite the model and checkpoint.

    Parameters
    ----------
    model: nn.Module

    criterion: nn.Module

    dataset: str
        Name of dataset.

    chckpt_name: str
        Name of the checkpoint file.

    is_retrain: bool, optional
        Whether to retrain the model and save it, instead of loading a pretraine one.

    transformer: sklearn.Transformer, optional
        Preprocesser to apply to the input before the loaded model. If not `None`
        the output will be a sklearn pipeline. If `not is_retrain` should already
        be fitted.

    basedir: str, optional
        Base directory where the checpoint will be saved.

    seed: int, optional
        Random seed for deterministic training.

    max_epochs: int, optional
        Maximum number of epochs.

    kwargs:
        Additional arguments to NeuralNetEstimator
    """
    chckpt_name = os.path.join(basedir, chckpt_name)

    if is_all_labels:
        chckpt_name += "_is_all_labels"

    if is_retrain:
        train, devset, test = get_train_dev_test_ssl(dataset, is_all_labels=is_all_labels)

        if transformer is not None and devset is not None:
            # given devset should also be transformed
            X_transf = transformer.transform(devset)
            devset = skorch.dataset.Dataset(X_transf, devset.targets)
    else:
        devset = None

    chckpt_name += suffix

    chckpt = Checkpoint(dirname=chckpt_name)

    callbacks = [ProgressBar(),
                 chckpt,
                 TrainEndCheckpoint(dirname=chckpt_name, fn_prefix='train_end_'),
                 FixRandomSeed(seed)]

    if devset is not None and patience is not None:
        callbacks.append(EarlyStopping(patience=patience))

    net = Trainer(model,
                  criterion=criterion,
                  callbacks=callbacks,
                  devset=devset,
                  max_epochs=max_epochs,
                  **kwargs)

    if is_retrain:
        if transformer is not None:
            net = Pipeline([('m1', transformer), ('classifier', net)])
        net.fit(train, train.targets)
    else:
        net.initialize()
        net.load_params(checkpoint=chckpt)
        if transformer is not None:
            net = Pipeline([('m1', transformer), ('classifier', net)])

    return net


def _DeepMLP(*args):
    return MLP(*args, n_hidden_layers=3)


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


def _preprocess(v):
    if v == "None":
        return None
    return v


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

    # for simplicity in command line can also give "None" instead of None
    args.kwargs = {k: _preprocess(v) for k, v in args.kwargs.items()}
    load_pretrained(args.model, args.dataset, is_retrain=True, **args.kwargs)
