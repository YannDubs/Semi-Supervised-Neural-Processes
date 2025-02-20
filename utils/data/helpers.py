import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.model_selection import PredefinedSplit

UNLABELLED_CLASS = -1


def merge_train_dev(train, dev):
    """
    Merge the train and dev `skorch.Dataset` and return the associated
    `sklearn.model_selection.PredefinedSplit`.
    """
    train_valid_X = np.concatenate((train.X, dev.X))
    train_valid_y = np.concatenate((train.y, dev.y))
    cv = PredefinedSplit([-1 for _ in range(len(train))
                          ] + [0 for _ in range(len(dev))])
    return train_valid_X, train_valid_y, cv


def make_ssl_dataset_(supervised, n_labels,
                      unlabeled_class=UNLABELLED_CLASS, seed=123, is_stratify=True):
    """Take a supervised dataset and turn it into an unsupervised one(inplace),
    by giving a special unlabeled class as target."""
    stratify = supervised.targets if is_stratify else None
    idcs_unlabel, indcs_labels = train_test_split(list(range(len(supervised))),
                                                  stratify=stratify,
                                                  test_size=n_labels,
                                                  random_state=seed)

    supervised.n_labels = len(indcs_labels)
    # cannot set _DatasetSubset through indexing
    targets = supervised.targets
    targets[idcs_unlabel] = unlabeled_class
    supervised.targets = targets


class _DatasetSubset(Dataset):
    """Helper to split train dataset into train and dev dataset.

    Parameters
    ----------
    to_split: Dataset
        Dataset to subset.

    idx_mapping: array-like
        Indices of the subset.

    Notes
    -----
    - Modified from: https: // gist.github.com / Fuchai / 12f2321e6c8fa53058f5eb23aeddb6ab
    - Does modify the length and targets with indexing anymore! I.e.
    `d.targets[1]=-1` doesn't work because np.array doesn't allow `arr[i][j]=-1`
    but you can do `d.targets=targets`
    """

    def __init__(self, to_split, idx_mapping):
        self.idx_mapping = idx_mapping
        self.length = len(idx_mapping)
        self.to_split = to_split

    def __getitem__(self, index):
        return self.to_split[self.idx_mapping[index]]

    def __len__(self):
        return self.length

    @property
    def targets(self):
        return self.to_split.targets[self.idx_mapping]

    @targets.setter
    def targets(self, values):
        self.to_split.targets[self.idx_mapping] = values

    @property
    def data(self):
        return self.to_split.data[self.idx_mapping]

    def __getattr__(self, attr):
        return getattr(self.to_split, attr)


def train_dev_split(to_split, dev_size=0.1, seed=123, is_stratify=True):
    """Split a training dataset into a training and validation one.

    Parameters
    ----------
    dev_size: float or int
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the dev split. If int, represents the absolute
        number of dev samples.

    seed: int
        Random seed.

    is_stratify: bool
        Whether to stratify splits based on class label.
    """
    n_all = len(to_split)
    idcs_all = list(range(n_all))
    stratify = to_split.targets if is_stratify else None
    idcs_train, indcs_val = train_test_split(idcs_all,
                                             stratify=stratify,
                                             test_size=dev_size,
                                             random_state=seed)
    train = _DatasetSubset(to_split, idcs_train)
    valid = _DatasetSubset(to_split, indcs_val)

    return train, valid
