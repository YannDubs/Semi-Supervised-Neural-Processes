import numpy as np
import skorch
from skorch.callbacks.scoring import check_scoring

import omegaconf
from omegaconf import OmegaConf

from wildml.training.trainer import _single_epoch


def to_numpy(X):
    """Generic function to convert array like to numpy."""
    if isinstance(X, list):
        X = np.array(X)
    return skorch.utils.to_numpy(X)


def get_exponential_decay_gamma(scheduling_factor, max_epochs):
    """Return the exponential learning rate factor gamma.

    Parameters
    ----------
    scheduling_factor :
        By how much to reduce learning rate during training.

    max_epochs : int
        Maximum number of epochs.
    """
    return (1 / scheduling_factor) ** (1 / max_epochs)


def hyperparam_to_path(hyperparameters):
    """Return a string of all hyperparameters that can be used as a path extension."""
    return "/".join([f"{k}_{v}" for k, v in hyperparameters.items()])


def format_container(to_format, formatter, k=None):
    """Format a container of string.

    Parameters
    ----------
    to_format : str, list, dict, or omegaconf.Config
        (list of) strings to fromat.
    
    formatter : dict
        Dict of keys to replace and values with which to replace.
    """
    if isinstance(to_format, str):
        out = to_format.format(**formatter)
    else:
        if isinstance(to_format, omegaconf.Config):
            to_format = OmegaConf.to_container(to_format, resolve=True)

        if isinstance(to_format, list):
            out = [format_container(path, formatter, k=i) for i, path in enumerate(to_format)]
        elif isinstance(to_format, dict):
            out = {k: format_container(path, formatter, k=k) for k, path in to_format.items()}
        else:
            raise ValueError(f"Unkown to_format={to_format}")
    return out


# Change _scoring for computing validation only at certain epochs
def _scoring(self, net, X_test, y_test):
    """Resolve scoring and apply it to data. Use cached prediction
        instead of running inference again, if available."""
    scorer = check_scoring(net, self.scoring_)

    if y_test is None:
        return float("nan")  #! Only difference : make sure no issue if valid not computed

    return scorer(net, X_test, y_test)


def _single_epoch_skipvalid(
    self,
    dataset,
    training,
    epoch,
    save_epochs=(
        list(range(10))
        + list(range(9, 100, 10))
        + list(range(99, 1000, 50))
        + list(range(999, 10000, 500))
    ),
    **fit_params,
):
    if not training and epoch not in save_epochs:
        return

    _single_epoch(self, dataset, training, epoch, **fit_params)
