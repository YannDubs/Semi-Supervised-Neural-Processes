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


from .helpers import FixRandomSeed, make_Xy_input
from .loaders import get_ssl_iterator

__all__ = ["NeuralNetEstimator", "NeuralNetTransformer", "NeuralNetClassifier"]


def get_doc(wrapped, add_methods="", add_param="", add_dflts=""):

    return """Wrapper around `{}`. Differences:

    Methods
    -------
    freeze:
        Freezes the model such that it cannot be fitted again.
    {}

    Parameters
    ----------
    devset: Dataset, optional
        Development set. If not `None` doesn't use cross validation but only
        this set.

    seed: int, optional
        Random seed or `None`.
    {}

    Defaults
    --------
    - device="gpu" which uses gpu if available
    - callbacks=[ProgressBar]
    - optimizer=Adam,
    - lr=0.001,
    - warm_start=True,
    - batch_size=128,
    - iterator_train__shuffle=True
    - iterator_valid__batch_size=128
    {}

    Base documentation:

    """.format(wrapped, add_methods, add_param, add_dflts) + wrapped.__doc__


class NeuralNetEstimator(skorch.NeuralNet):
    __doc__ = get_doc(skorch.NeuralNet)

    def __init__(self, *args,
                 devset=None,
                 seed=123,
                 device="gpu",
                 callbacks=[ProgressBar()],
                 optimizer=Adam,
                 lr=0.001,
                 warm_start=True,
                 batch_size=128,
                 **kwargs):

        self._is_frozen = False

        # uses cuda if available by defualt
        if device == "gpu":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if devset is not None:
            assert "train_split" not in kwargs
            kwargs["train_split"] = predefined_split(devset)

        if "iterator_train" not in kwargs and "iterator_train__shuffle" not in kwargs:
            # only default to shuffle when using default iterator
            kwargs["iterator_train__shuffle"] = True

        if "iterator_valid" not in kwargs and "iterator_valid__batch_size" not in kwargs:
            # only default to shuffle when using default iterator
            kwargs["iterator_valid__batch_size"] = 128

        callbacks += [FixRandomSeed(seed)]

        super().__init__(*args, callbacks=callbacks, device=device, optimizer=optimizer,
                         lr=lr, warm_start=warm_start, batch_size=batch_size, **kwargs)

    def freeze(self, is_freeze=True):
        """Freezes (or unfreeze) the model such that it cannot be fitted again."""
        self._is_frozen = is_freeze
        return self

    def fit(self, X, y=None, **fit_params):
        if self._is_frozen:
            if self.verbose > 0:
                warnings.warn("Skipping fitting because froze etimator.")
            return self

        return super().fit(X, y=y, **fit_params)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        # same but tries to redirect X
        y_true = to_tensor(y_true, device=self.device)
        if X is not None:
            X = to_tensor(X, device=self.device)
        try:
            loss = self.criterion_(y_pred, y_true, X=X)
        except TypeError:
            loss = self.criterion_(y_pred, y_true)
        return loss


is_ssl_doc = """
is_ssl: bool, optional
    Wether the estimator is semi supervised. I.e. `forward` method of
    the model takes a `y` as input. The batches will include 50%
    labelled and 50% unlabelled. When `True` a dataset needs to be
    given to `.fit` and it needs to have a `target` or `y` attribute with
    -1 when unlabelled.
    """


# careful : diamond inheritance
class NeuralNetClassifier(NeuralNetEstimator, skorch.NeuralNetClassifier, ClassifierMixin):
    __doc__ = get_doc(skorch.NeuralNetClassifier, add_param=is_ssl_doc)

    def __init__(self, *args,
                 devset=None,
                 criterion=torch.nn.CrossEntropyLoss,
                 is_ssl=True,
                 **kwargs):

        self.is_ssl = is_ssl
        if self.is_ssl:
            if devset is not None:
                devset = Dataset(*make_Xy_input(devset))

            assert "iterator_train" not in kwargs
            kwargs["iterator_train"] = get_ssl_iterator

        # diamond structure => first call NeuralNetEstimator => take its hyperparam
        super().__init__(*args, criterion=criterion, devset=devset,
                         **kwargs)

    # both parent classes redefine this method (diamond)
    get_loss = NeuralNetEstimator.get_loss

    def fit(self, X, y=None, **fit_params):
        if self.is_ssl:
            return skorch.NeuralNet.fit(self, *make_Xy_input(X, y=y), **fit_params)
        return NeuralNetEstimator.fit(self, X, y=y, **fit_params)

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


transform = """
    transform:
        Ridirect to `.predict`.
    """


class NeuralNetTransformer(NeuralNetEstimator, TransformerMixin):
    __doc__ = get_doc(skorch.NeuralNet, add_methods=transform)

    def transform(self, X):
        """Transform an input."""
        self.module_.is_transform = True
        X_transf = []
        for out in self.forward_iter(X, training=False):
            out = out[0] if isinstance(out, tuple) else out
            X_transf.append(to_numpy(out))
        X_transf = np.concatenate(X_transf, 0)
        self.module_.is_transform = False
        return X_transf
