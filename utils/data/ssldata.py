import os
import logging

from PIL import Image
import numpy as np
import sklearn.datasets
import joblib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

import skorch.dataset
from skorch.utils import to_numpy

from utils.helpers import set_attributes
from .helpers import train_dev_split, make_ssl_dataset_
from .transforms import (precompute_batch_tranforms, global_contrast_normalization,
                         zca_whitening, random_translation, robust_minmax_scale,
                         gaussian_noise)
from .tsdata import get_timeseries_dataset

DIR = os.path.abspath(os.path.dirname(__file__))

DATASETS_DICT = {"cifar10": "CIFAR10",
                 "svhn": "SVHN",
                 "mnist": "MNIST",
                 "pts_circles": "None",
                 "pts_moons": "None",
                 "pts_var_gaus": "None",
                 "pts_cov_gaus": "None",
                 "pts_iso_gaus": "None"}
DATASETS = list(DATASETS_DICT.keys())
N_LABELS = {"cifar10": 4000,
            "svhn": 1000,
            "mnist": 100,
            "pts_circles": 10,
            "pts_moons": 6,
            "pts_var_gaus": 12,
            "pts_cov_gaus": 12,
            "pts_iso_gaus": 12}


def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        try:
            return get_timeseries_dataset(dataset)
        except KeyError:
            raise ValueError("Unkown dataset: {}".format(dataset))


def get_train_dev_test_ssl(dataset,
                           n_labels=None,
                           root=None,
                           dev_size=0.1,
                           seed=123,
                           **kwargs):
    """Return the training, validation and test dataloaders

    Parameters
    ----------
    dataset : {"cifar", "svhn"}
        Name of the dataset to load

    n_labels : int
        Number of labels to keep. If `None` uses dataset specific default.

    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    dev_size : float or int, optional
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the dev split. If int, represents the absolute
        number of dev samples.

    seed : int, optional
        Random seed.

    kwargs :
        Additional arguments to `generate_train_dev_test_ssl`.
    """
    _Dataset = get_dataset(dataset)

    if n_labels is None:
        n_labels = N_LABELS[dataset]

    if _Dataset is None:
        # has to generate
        return generate_train_dev_test_ssl(dataset, n_labels,
                                           dev_size=dev_size,
                                           seed=seed,
                                           **kwargs)

    if root is not None:
        kwargs["root"] = root
    # important to do train before test => compute ZCA on train
    train = _Dataset(split="train", **kwargs)
    test = _Dataset(split="test", **kwargs)

    # Nota Bene: we are actually first doing the transformatiosn such as GCN
    # the dev splitting => normalization also done on dev, which is not ideal
    # but fine because not doing on the test set => final results will be correct
    # but hyperparametrization is a little biased
    if dev_size > 0:
        train, dev = train_dev_split(train, dev_size=dev_size, seed=seed, is_stratify=True)
    else:
        dev = None
    make_ssl_dataset_(train, n_labels, seed=seed, is_stratify=True)

    return train, dev, test


# POINT DATASETS
def generate_train_dev_test_ssl(dataset, n_label,
                                n_unlabel=int(1e4),
                                n_test=int(1e4),
                                dev_size=0.1, seed=123, is_hard=False):
    """Genererate simple ssl datasets.

    Parameters
    ----------
    dataset : {"pts_circles", "pts_moons", "pts_var_gaus", "pts_cov_gaus", "pts_iso_gaus"}
        Name of the dataset to generate

    n_labels : int
        Number of labelled examples.

    n_lunabels : int
        Number of unlabelled examples.

    n_test : int
        Number of test examples.

    root : str
        Path to the dataset root. If `None` uses the default one.

    dev_size : float or int
        If float, should be between 0.0 and 1.0 and represent a ratio beteen the
        n_unlabel and size of the dev set. If int, represents the absolute number
        of dev samples.

    seed : int, optional
        Random seed.

    is_hard : bool, optional
        Whether to increase nosie / variance by 1.5x to make the task more difficult.
    """
    n_dev = int(n_unlabel * dev_size) if dev_size < 1 else dev_size
    hard_factor = 1.5 if is_hard else 1  # multiply by 1.5 if hard

    gaus_means = np.array([[7, 9], [8., -1], [-5., -9.]])
    args = dict(pts_circles={"f": sklearn.datasets.make_circles,
                             "std": 0.09, "kwargs": dict(factor=.5)},
                pts_moons={"f": sklearn.datasets.make_moons,
                           "std": 0.13, "kwargs": {}},
                pts_iso_gaus={"f": sklearn.datasets.make_blobs,
                              "std": 1.5, "kwargs": dict(centers=gaus_means)},
                pts_cov_gaus={"f": sklearn.datasets.make_blobs,
                              "std": 1.5,
                              "kwargs": dict(centers=gaus_means)},
                pts_var_gaus={"f": sklearn.datasets.make_blobs,
                              "std": np.array([1.0, 2.5, 0.5]),
                              "kwargs": dict(centers=gaus_means)})

    spec_args = args[dataset]

    def get_noisy_kwargs(is_label=False):
        if "_gaus" in dataset:
            std_label_factor = 1 if not is_label else 0.5  # divide by 10 std
            spec_args["kwargs"]["cluster_std"] = spec_args["std"] * std_label_factor * hard_factor
        else:
            std_label_factor = 1 if not is_label else 0  # no noise
            spec_args["kwargs"]["noise"] = spec_args["std"] * std_label_factor * hard_factor
        return spec_args["kwargs"]

    X_lab, y_lab = spec_args["f"](n_samples=n_label, random_state=seed,
                                  **get_noisy_kwargs(True))
    X_unlab, y_unlab = spec_args["f"](n_samples=n_unlabel, random_state=seed,
                                      **get_noisy_kwargs())
    y_unlab[:] = -1
    X_train = np.concatenate([X_lab, X_unlab])
    y_train = np.concatenate([y_lab, y_unlab])
    X_dev, y_dev = spec_args["f"](n_samples=n_dev, random_state=seed,
                                  **get_noisy_kwargs())
    X_test, y_test = spec_args["f"](n_samples=n_test, random_state=seed,
                                    **get_noisy_kwargs())

    if dataset == "pts_cov_gaus":
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_train = np.dot(X_train, transformation)
        X_dev = np.dot(X_dev, transformation)
        X_test = np.dot(X_test, transformation)

    # set a shape and n_classes for compatibility
    att = dict(shape=(2,), n_classes=3)
    return (set_attributes(skorch.dataset.Dataset(X_train.astype(np.float32), y=y_train), **att),
            set_attributes(skorch.dataset.Dataset(X_dev.astype(np.float32), y=y_dev), **att),
            set_attributes(skorch.dataset.Dataset(X_test.astype(np.float32), y=y_test), **att))


# IMAGE DATASETS
class CIFAR10(datasets.CIFAR10):
    """CIFAR10 wrapper. Docs: `datasets.CIFAR10.`

    Notes
    -----
    - Transformations (and their order) follow [1] besides the fact that we scale
    the images to be in [0,1] to make it easier to use probabilistic generative models.

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test'}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.CIFAR10`.

    References
    ----------
    [1] Oliver, A., Odena, A., Raffel, C. A., Cubuk, E. D., & Goodfellow, I.
        (2018). Realistic evaluation of deep semi-supervised learning algorithms.
        In Advances in Neural Information Processing Systems (pp. 3235-3246).
    """
    shape = (3, 32, 32)
    n_classes = 10

    def __init__(self,
                 root=os.path.join(DIR, '../../data/CIFAR10'),
                 split="train",
                 logger=logging.getLogger(__name__),
                 **kwargs):

        if split == "train":
            transforms_list = [transforms.RandomHorizontalFlip(),
                               transforms.Lambda(lambda x: random_translation(x, 2)),
                               transforms.ToTensor(),
                               # adding random noise of std 0.15 but clip to 0,1
                               transforms.Lambda(lambda x: gaussian_noise(x, std=0.15 * self.noise_factor, min=0, max=1))]
        elif split == "test":
            transforms_list = [transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(root,
                         train=split == "train",
                         download=True,
                         transform=transforms.Compose(transforms_list),
                         **kwargs)

        basename = os.path.join(root, "clean_{}".format(split))
        transforms_X = [global_contrast_normalization,
                        lambda x: zca_whitening(x, root, is_load=split == "test"),
                        lambda x: robust_minmax_scale(x, root, is_load=split == "test"),
                        lambda x: (x * 255).astype(np.uint8)]  # back to 255 for PIL
        self.data, self.targets = precompute_batch_tranforms(self.data, self.targets, basename,
                                                             transforms_X=transforms_X,
                                                             logger=logger)
        self.targets = to_numpy(self.targets)

        # DIRTY make sure that the noise added is also scaled
        robust_scaler = joblib.load(os.path.join(root, "robust_scaler.npy"))
        self.noise_factor = 1 / robust_scaler.data_range_[0]


class SVHN(datasets.SVHN):
    """SVHN wrapper. Docs: `datasets.SVHN.`

    Notes
    -----
    - Transformations (and their order) follow [1] besides the fact that we scale
    the images to be in [0,1] isntead of [-1,1] to make it easier to use
    probabilistic generative models.

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test', "extra"}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.CIFAR10`.

    References
    ----------
    [1] Oliver, A., Odena, A., Raffel, C. A., Cubuk, E. D., & Goodfellow, I.
        (2018). Realistic evaluation of deep semi-supervised learning algorithms.
        In Advances in Neural Information Processing Systems (pp. 3235-3246).
    """
    shape = (3, 32, 32)
    n_classes = 10

    def __init__(self,
                 root=os.path.join(DIR, '../../data/SVHN'),
                 split="train",
                 logger=logging.getLogger(__name__),
                 **kwargs):

        if split == "train":
            transforms_list = [transforms.Lambda(lambda x: random_translation(x, 2)),
                               transforms.ToTensor()]
        elif split == "test":
            transforms_list = [transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(root,
                         split=split,
                         download=True,
                         transform=transforms.Compose(transforms_list),
                         **kwargs)

        self.labels = to_numpy(self.labels)

    @property
    def targets(self):
        # make compatible with CIFAR10 dataset
        return self.labels


class MNIST(datasets.MNIST):
    """MNIST wrapper. Docs: `datasets.MNIST.`

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test', "extra"}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.MNIST`.
    """
    shape = (1, 32, 32)
    n_classes = 10

    def __init__(self,
                 root=os.path.join(DIR, '../../data/MNIST'),
                 split="train",
                 logger=logging.getLogger(__name__),
                 **kwargs):

        if split == "train":
            transforms_list = [transforms.Resize(32),
                               transforms.ToTensor()]
        elif split == "test":
            transforms_list = [transforms.Resize(32),
                               transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(root,
                         train=split == "train",
                         download=True,
                         transform=transforms.Compose(transforms_list),
                         **kwargs)

        self.targets = to_numpy(self.targets)
