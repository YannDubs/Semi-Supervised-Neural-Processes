import warnings
from contextlib import suppress

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import TransformerMixin, ClassifierMixin
from scipy.special import softmax

import skorch
from skorch.callbacks import ProgressBar, EpochScoring
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from skorch.utils import to_numpy, to_tensor
from skorch.dataset import uses_placeholder_y, unpack_data, get_len


from .helpers import FixRandomSeed, make_Xy_input
from .loaders import get_ssl_iterator

__all__ = ["NeuralNetTransformer", "NeuralNetClassifier"]


def get_loss(self, y_pred, y_true, X=None, training=False):
    """Return the loss for this batch."""
    y_true = to_tensor(y_true, device=self.device)

    if isinstance(self.criterion_, nn.Module):
        if training:
            self.criterion_.train()
        else:
            self.criterion_.eval()

    return self.criterion_(y_pred, y_true)


def fit_loop(self, X, y=None, epochs=None, **fit_params):

    self.check_data(X, y)
    epochs = epochs if epochs is not None else self.max_epochs

    dataset_train, dataset_valid = self.get_split_datasets(X, y, **fit_params)
    on_epoch_kwargs = {"dataset_train": dataset_train, "dataset_valid": dataset_valid}

    for epoch in range(epochs):
        self.notify("on_epoch_begin", **on_epoch_kwargs)

        self._single_epoch(dataset_train, training=True, epoch=epoch, **fit_params)

        if dataset_valid is not None:
            self._single_epoch(dataset_valid, training=False, epoch=epoch, **fit_params)

        self.notify("on_epoch_end", **on_epoch_kwargs)

    return self


def _single_epoch(self, dataset, training, epoch, **fit_params):
    """Computes a single epoch of train or validation."""
    is_placeholder_y = uses_placeholder_y(dataset)

    if training:
        prfx = "train"
        step_fn = self.train_step
    else:
        prfx = "valid"
        step_fn = self.validation_step

    batch_count = 0
    for data in self.get_iterator(dataset, training=training):
        Xi, yi = unpack_data(data)
        yi_res = yi if not is_placeholder_y else None
        self.notify("on_batch_begin", X=Xi, y=yi_res, training=training)
        step = step_fn(Xi, yi, **fit_params)
        self.history.record_batch(prfx + "_loss", step["loss"].item())
        self.history.record_batch(prfx + "_batch_size", get_len(Xi))
        self.notify("on_batch_end", X=Xi, y=yi_res, training=training, **step)
        batch_count += 1

    self.history.record(prfx + "_batch_count", batch_count)

    if hasattr(self.criterion_, "to_store"):
        for k, v in self.criterion_.to_store.items():
            with suppress(NotImplementedError):
                # pytorch raises NotImplementedError on wrong types
                self.history.record(prfx + "_" + k, (v[0] / v[1]).item())
        self.criterion_.to_store = dict()


doc_neural_net_clf = (
    """Wrapper around skorch.NeuralNetClassifier to enable semi supervised learning. Differences:

    Parameters
    ----------
    is_ssl: bool, optional
        Wether the estimator is semi supervised. I.e. `forward` method of
        the model takes a `y` as input. The batches will include 50%
        labelled and 50% unlabelled. When `True` a dataset needs to be
        given to `.fit` and it needs to have a `target` or `y` attribute with
        -1 when unlabelled.

    Notes
    -----
    - use by default crossentropy loss instead of NNLoss
    - enables storing of additional losses.

    Base documentation:
    """
    + skorch.NeuralNetClassifier.__doc__
)


class NeuralNetClassifier(skorch.NeuralNetClassifier):

    __doc__ = doc_neural_net_clf

    def __init__(self, *args, criterion=torch.nn.CrossEntropyLoss, is_ssl=False, **kwargs):
        super().__init__(*args, criterion=criterion, **kwargs)

        self.is_ssl = is_ssl
        if self.is_ssl:
            assert "iterator_train" not in kwargs
            kwargs["iterator_train"] = get_ssl_iterator

    def fit(self, X, y=None, **fit_params):
        if self.is_ssl:
            # gives y as input like that can treat differently the ones that are sup or not
            return super().fit(*make_Xy_input(X, y=y), **fit_params)
        return super().fit(X, y=y, **fit_params)

    def predict_proba(self, X):
        """Return probability estimates for samples.

        Notes
        -----
        - output of model should be logits (softmax applied in this function)
        - If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using :func:`~skorch.NeuralNet.forward`
        instead.

        Returns
        -------
        y_proba : numpy ndarray
        
        """
        # output of model should be logits!
        logits = super().predict_proba(X)
        return softmax(logits, axis=1)

    fit_loop = fit_loop
    _single_epoch = _single_epoch
    get_loss = get_loss


doc_neural_net_trnsf = (
    """Wrapper around skorch.NeuralNet for transforming data. Differences:

    Parameters
    ----------
    is_ssl: bool, optional
        Wether the estimator is semi supervised. I.e. `forward` method of
        the model takes a `y` as input. The batches will include 50%
        labelled and 50% unlabelled. When `True` a dataset needs to be
        given to `.fit` and it needs to have a `target` or `y` attribute with
        -1 when unlabelled.

    Methods
    -------
    freeze:
        Freezes the model such that it cannot be fitted again.

    transform:
        Returns a numpy array containing all the first outputs, similarly to `.predict`. The main 
        difference is that it sets `module_.is_transform` to `True`. The correct behavior should thus
        be implemented in the module using `is_transform` flag.

    Notes
    -----
    - enables storing of additional losses.

    Base documentation:
    """
    + skorch.NeuralNet.__doc__
)


class NeuralNetTransformer(skorch.NeuralNet, TransformerMixin):
    __doc__ = doc_neural_net_trnsf

    def freeze(self, is_freeze=True):
        """Freezes (or unfreeze) the model such that it cannot be fitted again."""
        self._is_frozen = is_freeze
        return self

    def fit(self, X, y=None, **fit_params):
        if hasattr(self, "_is_frozen") and self._is_frozen:
            if self.verbose > 0:
                warnings.warn("Skipping fitting because froze etimator.")
            return self

        return super().fit(X, y=y, **fit_params)

    def transform(self, X):
        """Transform an input."""
        self.module_.is_transform = True
        self.module_.training = False
        X_transf = self.predict(X)
        self.module_.is_transform = False
        self.module_.training = True
        return X_transf

    fit_loop = fit_loop
    _single_epoch = _single_epoch
    get_loss = get_loss

