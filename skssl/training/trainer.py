import warnings

import numpy as np
import torch
from torch.optim import Adam


from sklearn.base import TransformerMixin, ClassifierMixin
from scipy.special import softmax

import skorch
from skorch.callbacks import ProgressBar, EpochScoring
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from skorch.utils import to_numpy, to_tensor


from .helpers import FixRandomSeed, make_ssl_input
from .loaders import get_ssl_iterator

__all__ = ["NeuralNetEstimator","NeuralNetClassifier"]

NEW_DOC = """Wrapper around `{}`. Differences:

    Methods
    -------
    transform:
        That is the same as `.predict` but uses the second outputed value if
        multiple outputs and enables the use of neural networks as transformers
        in Sklearn

    freeze:
        Freezes the model such that it cannot be fitted again.


    Parameters
    ----------
    is_ssl: bool, optional
        Wether the estimator is semi supervised. I.e. `forward` method of
        the model takes a `y` as input. The batches will include 50%
        labelled and 50% unlabelled. When `True` a dataset needs to be
        given to `.fit` and it needs to have a `target` or `y` attribute with
        -1 when unlabelled.

    devset: Dataset, optional
        Development set. If not `None` doesn't use cross validation but only
        this set.

    seed: int, optional
        Random seed or `None`.

    Defaults
    --------
        - optimizer
        - warm_start=True
        - callbacks=[ProgressBar]
        - device="gpu" which uses gpu if available
        - iterator_train__shuffle=True
        - iterator_valid__batch_size=256

    Base documentation:

    """


class NeuralNetEstimator(skorch.NeuralNet, TransformerMixin):
    __doc__ = NEW_DOC.format("skorch.NeuralNet") + skorch.NeuralNet.__doc__

    def __init__(self, *args,
                 devset=None,
                 is_ssl=True,
                 seed=123,
                 device="gpu",
                 callbacks=[ProgressBar()],
                 optimizer=Adam,
                 lr=0.001,
                 warm_start=True,
                 iterator_valid__batch_size=256,
                 **kwargs):
        # uses cuda if available by defualt
        self.is_ssl = is_ssl
        self._is_frozen = False

        if device == "gpu":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.is_ssl:
            if devset is not None:
                devset = Dataset(*make_ssl_input(devset))

            assert "iterator_train" not in kwargs
            kwargs["iterator_train"] = get_ssl_iterator
        else:
            kwargs.update({"iterator_train__shuffle": False})

        if devset is not None:
            assert "train_split" not in kwargs
            kwargs["train_split"] = predefined_split(devset)

        # directly use parent because want to copy the constructor to NeuralNetCLassifier
        # note that this means TransformerMixin.__init__ is not called
        # which is fine as empty
        skorch.NeuralNet.__init__(self, *args,
                                  device=device,
                                  callbacks=callbacks + [FixRandomSeed(seed)],
                                  optimizer=optimizer,
                                  lr=lr,
                                  warm_start=warm_start,
                                  iterator_valid__batch_size=iterator_valid__batch_size,
                                  **kwargs)

    def freeze(self, is_freeze=True):
        """Freezes (or unfreeze) the model such that it cannot be fitted again."""
        self._is_frozen = is_freeze
        return self

    def fit(self, X, y=None, **fit_params):
        if self._is_frozen:
            if self.verbose > 0:
                warnings.warn("Skipping fitting because froze etimator.")
            return self

        if self.is_ssl:
            return skorch.NeuralNet.fit(self, *make_ssl_input(X, y=y), **fit_params)

        return skorch.NeuralNet.fit(self, X, y=y, **fit_params)

    def transform(self, X):
        """Same as `predict` for NeuralNet but uses the second value if multiple outputs."""
        X_transformed = []
        for preds in self.forward_iter(X, training=False):
            preds = preds[1] if isinstance(preds, tuple) else preds
            X_transformed.append(to_numpy(preds))
        X_transformed = np.concatenate(X_transformed, 0)
        return X_transformed

    def get_loss(self, y_pred, y_true, X=None, training=False):
        # same but tries to redirect X
        y_true = to_tensor(y_true, device=self.device)
        try:
            loss = self.criterion_(y_pred, y_true, X=X)
        except TypeError:
            loss = self.criterion_(y_pred, y_true)
        return loss

    def validation_step(self, Xi, yi, **fit_params):
        # Make sure that Xi is on device
        Xi = to_tensor(Xi, device=self.device)
        return skorch.NeuralNet.validation_step(self, Xi, yi, **fit_params)

    def train_step_single(self, Xi, yi, **fit_params):
        # Make sure that Xi is on device
        Xi = to_tensor(Xi, device=self.device)
        return skorch.NeuralNet.train_step_single(self, Xi, yi, **fit_params)


class NeuralNetClassifier(skorch.NeuralNetClassifier, ClassifierMixin):
    __doc__ = NEW_DOC.format("skorch.NeuralNetClassifier") + skorch.NeuralNetClassifier.__doc__
    __init__ = NeuralNetEstimator.__init__
    freeze = NeuralNetEstimator.freeze
    fit = NeuralNetEstimator.fit
    transform = NeuralNetEstimator.transform
    validation_step = NeuralNetEstimator.validation_step
    train_step_single = NeuralNetEstimator.train_step_single
    get_loss = NeuralNetEstimator.get_loss

    def predict_proba(self, X):
        """Where applicable, return probability estimates for
        samples.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        Returns
        -------
        y_proba : numpy ndarray
        """
        logits = super().predict_proba(X)
        return softmax(logits, axis=1)
