import os
import glob
import logging

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image


from utils.helpers import to_numpy

from .base import BaseDataset
from .helpers import get_masks_drop_features


COLOUR_BLACK = torch.tensor([0.0, 0.0, 0.0])
COLOUR_WHITE = torch.tensor([1.0, 1.0, 1.0])
COLOUR_BLUE = torch.tensor([0.0, 0.0, 1.0])
DATASETS_DICT = {"mnist": "MNIST", "cifar10": "CIFAR10", "cifar100": "CIFAR100", "svhn": "SVHN"}
DATASETS = list(DATASETS_DICT.keys())


# HELPERS
def get_Dataset(dataset):
    """Return the correct uninstantiated datasets."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_Dataset(dataset).shape


class ImgDataset(BaseDataset):
    """Image dataset wrapper that adds nice functionalitites.
    
    Parameters
    ----------
    is_augment : bool, optional
        Whether to transform the training set (and thus validation).

    split : {'train', 'test', ...}, optional
        According dataset is selected.
    """

    is_numbers = False

    def __init__(self, *args, is_augment=True, split="train", **kwargs):
        super().__init__(*args, **kwargs)

        self.is_augment = is_augment
        self.split = split
        self.is_drop_features = False  # by default return all features

        if self.is_augment and self.split == "train":
            self.set_train_transforms()
        else:
            self.set_test_transforms()

        if self.is_random_targets:
            self.randomize_targets_()

    def set_train_transforms(self):
        """Return the training transformation."""
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.shape[1], self.shape[2])),
                # the following performs translation
                transforms.RandomCrop((self.shape[1], self.shape[2]), padding=4),
                # don't flip if working with numbers
                transforms.RandomHorizontalFlip() if not self.is_numbers else torch.nn.Identity(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def set_test_transforms(self):
        """Return the testing transformation."""
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.shape[1], self.shape[2])),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def rm_transformations(self):
        """Completely remove transformation. Used to plot or compute mean and variance."""
        self.transform = transforms.Compose([transforms.ToTensor()])

    def make_test(self):
        """Make the data a test set."""
        self.set_test_transforms()

    def drop_features_(self, drop_size):
        """Drop part of the features (pixels in images).

        Note
        ----
        - this function actually just precomputes the `self.to_drop` of values that should be droped 
        the dropping is in `__get_item__`.

        Parameters
        ----------
        drop_size : float or int or tuple, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to 
            drop. If int, represents the number of datapoints to drop. If tuple, same as before 
            but give bounds (min and max). 0 means keep all.
        """
        self.logger.info(f"drop_features_ {drop_size} features...")

        assert not self.is_drop_features, "cannot drop multiple times the features"

        self.is_drop_features = True

        self.to_drop = get_masks_drop_features(
            drop_size, [self.shape[1], self.shape[2]], len(self), seed=self.seed
        )

    def __getitem__(self, index):
        X, target = super().__getitem__(index)

        target = self.add_index(target, index)

        if self.is_drop_features:
            X[:, self.to_drop[index]] = float("nan")

        return X, target


# TORCHVISION DATASETS
class SVHN(ImgDataset, datasets.SVHN):
    """SVHN wrapper. Docs: `datasets.SVHN.`

    Parameters
    ----------
    kwargs:
        Additional arguments to `ImgDataset`.

    Examples
    --------
    >>> data = SVHN(split="train") #doctest:+ELLIPSIS
    Using ...
    >>> len(data)
    73257
    >>> len(data) == len(data.data) == len(data.targets) 
    True
    >>> [type(i) for i in data[0]]
    [<class 'torch.Tensor'>, <class 'int'>]

    >>> train, valid = data.train_test_split(test_size=1000, is_stratify=True)
    >>> len(valid)
    1000

    >>> data.drop_labels_(0.9)
    >>> round(len([t for t in data.targets if t == -1]) / len(data), 1)
    0.9

    >>> data.balance_labels_()
    >>> len(data)
    131864
    >>> data.drop_unlabelled_()
    >>> len(data)
    65932

    >>> data.drop_features_(0.7)
    >>> round((torch.isnan(data[0][0])).float().mean().item(), 1)
    0.7
    >>> data[0][0][0] # showing image for one channel
    tensor([[    nan,     nan, -2.2102,  ...,     nan,     nan,     nan],
            [    nan,     nan,     nan,  ...,     nan,     nan,     nan],
            [    nan, -2.2102,     nan,  ...,     nan,     nan,     nan],
            ...,
            [-2.2102,     nan, -2.2102,  ..., -2.2102,     nan,     nan],
            [    nan,     nan,     nan,  ...,     nan,     nan,     nan],
            [    nan,     nan,     nan,  ...,     nan, -2.2102,     nan]])

    >>> data[0][1]
    1
    >>> data.randomize_targets_()
    >>> data[0][1]
    8
    """

    shape = (3, 32, 32)
    missing_px_color = COLOUR_BLACK
    n_classes = 10
    n_train = 73257
    mean = [0.43768448, 0.4437684, 0.4728041]
    std = [0.19803017, 0.20101567, 0.19703583]
    is_numbers = True

    def __init__(self, **kwargs):
        ImgDataset.__init__(self, **kwargs)
        datasets.SVHN.__init__(
            self, self.root, download=True, split=self.split, transform=self.transform
        )
        self.labels = to_numpy(self.labels)

    @property
    def targets(self):
        # make compatible with CIFAR10 dataset
        return self.labels

    @targets.setter
    def targets(self, values):
        self.labels = values


class CIFAR10(ImgDataset, datasets.CIFAR10):
    """CIFAR10 wrapper. Docs: `datasets.CIFAR10.`

    Parameters
    ----------
    kwargs:
        Additional arguments to `datasets.CIFAR10` and `ImgDataset`.

    Examples
    --------
    See SVHN
    """

    shape = (3, 32, 32)
    n_classes = 10
    missing_px_color = COLOUR_BLACK
    n_train = 50000
    mean = [0.4914009, 0.48215896, 0.4465308]
    std = [0.24703279, 0.24348423, 0.26158753]

    def __init__(self, **kwargs):

        ImgDataset.__init__(self, **kwargs)
        datasets.CIFAR10.__init__(
            self, self.root, download=True, train=self.split == "train", transform=self.transform
        )
        self.targets = to_numpy(self.targets)


class CIFAR100(ImgDataset, datasets.CIFAR100):
    """CIFAR100 wrapper. Docs: `datasets.CIFAR100.`

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test'}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.CIFAR100` and `ImgDataset`.

    Examples
    --------
    See SVHN
    """

    shape = (3, 32, 32)
    n_classes = 100
    n_train = 50000
    missing_px_color = COLOUR_BLACK
    mean = [0.5070754, 0.48655024, 0.44091907]
    std = [0.26733398, 0.25643876, 0.2761503]

    def __init__(self, **kwargs):

        ImgDataset.__init__(self, **kwargs)
        datasets.CIFAR100.__init__(
            self, self.root, download=True, train=self.split == "train", transform=self.transform
        )
        self.targets = to_numpy(self.targets)


class MNIST(ImgDataset, datasets.MNIST):
    """MNIST wrapper. Docs: `datasets.MNIST.`

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test', "extra"}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.MNIST` and `ImgDataset`.

    Examples
    --------
    See SVHN
    """

    shape = (1, 32, 32)
    n_classes = 10
    n_examples = 60000
    missing_px_color = COLOUR_BLUE
    mean = [0.13066062]
    std = [0.30810776]
    is_numbers = True

    def __init__(self, **kwargs):

        ImgDataset.__init__(self, **kwargs)
        datasets.MNIST.__init__(
            self, self.root, download=True, train=self.split == "train", transform=self.transform
        )
        self.targets = to_numpy(self.targets)


if __name__ == "__main__":
    import doctest
    import sys

    import os

    sys.path.append("../../")
    print(os.getcwd())
    doctest.testmod()
