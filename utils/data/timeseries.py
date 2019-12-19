import os
from subprocess import call

import numpy as np
import torch

from .base import BaseDataset
from .helpers import get_masks_drop_features


DIR = os.path.abspath(os.path.dirname(__file__))
DATASETS_DICT = {"har": "HARDataset"}
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


class TimeseriesDataset(BaseDataset):
    """Timeseries dataset wrapper that adds nice functionalitites.
    
    Parameters
    ----------
    split : {'train', 'test', ...}, optional
        According dataset is selected.
    """

    def __init__(self, *args, split="train", **kwargs):
        super().__init__(*args, **kwargs)
        self.is_drop_features = False  # make sure that drops only once for now
        self.split = split

    def drop_features_(self, drop_size):
        """Drop part of the features (multidimensional timeseries at a given time).

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
            drop_size, [self.data.size(1)], len(self), seed=self.seed
        )

    def __getitem__(self, index):
        X, target = super().__getitem__(index)

        target = self.add_index(target, index)

        return X, target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]

        if self.is_drop_features:
            X[self.to_drop[idx], :] = float("nan")

        return X, self.targets[idx]


# TODO: clean
class HARDataset(TimeseriesDataset):
    """Human Activity Recognition dataset. 

    Notes
    -----
    Credits : https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

    Parameters
    ----------
    kwargs : 
        Additional arguments to TimeseriesDataset.

    Examples
    --------
    >>> data = HARDataset(split="train")
    >>> len(data)
    7352
    >>> [type(i) for i in data[0]]
    [<class 'torch.Tensor'>, <class 'torch.Tensor'>]

    >>> train, valid = data.train_test_split(test_size=100, is_stratify=True)
    >>> len(valid)
    100

    >>> data.drop_labels_(0.9)
    >>> round(len([t for t in data.targets if t == -1]) / len(data), 1)
    0.9

    >>> data.balance_labels_()
    >>> len(data)
    13234
    >>> data.drop_unlabelled_()
    >>> len(data)
    6617

    >>> data.drop_features_(0.7)
    >>> round((torch.isnan(data[10][0])).float().mean().item(), 1)
    0.7
    >>> data[10][0]
    tensor([[    nan,     nan,     nan,  ...,     nan,     nan,     nan],
            [    nan,     nan,     nan,  ...,     nan,     nan,     nan],
            [ 0.0871, -0.0451,  0.1854,  ...,  0.4457,  2.0176,  1.6229],
            ...,
            [    nan,     nan,     nan,  ...,     nan,     nan,     nan],
            [ 0.0369,  0.0059,  0.0264,  ...,  0.4682,  2.0189,  1.6137],
            [    nan,     nan,     nan,  ...,     nan,     nan,     nan]])
    """

    n_classes = 6
    n_train = 7352

    TRAIN = "train/"
    TEST = "test/"

    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_",
    ]

    mean = [
        3.4212677e-09,
        4.8643618e-09,
        3.9563477e-09,
        -2.1403190e-09,
        6.1939538e-09,
        -3.5347694e-09,
        -7.3419429e-08,
        6.4858154e-09,
        -2.5424397e-08,
    ]

    std = [
        0.19484635,
        0.12242749,
        0.10687881,
        0.40681508,
        0.38185433,
        0.25574315,
        0.4141119,
        0.3909954,
        0.35776883,
    ]

    n_train = 7352
    n_total = 10299

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data_path = self.root + "UCI HAR Dataset/"
        self.download()
        self.load_all_split_()

        # standardize with the training mean and std
        self.data = (self.data - np.array(self.mean)) / np.array(self.std)

        self.data = torch.from_numpy(self.data).float()
        self.targets = torch.from_numpy(self.targets).long().squeeze()

    def load_all_split_(self):
        train_path = self.data_path + self.TRAIN
        test_path = self.data_path + self.TEST

        X_train_signals_paths = [
            train_path + "Inertial Signals/" + signal + "train.txt"
            for signal in self.INPUT_SIGNAL_TYPES
        ]
        X_test_signals_paths = [
            test_path + "Inertial Signals/" + signal + "test.txt"
            for signal in self.INPUT_SIGNAL_TYPES
        ]

        y_train_path = train_path + "y_train.txt"
        y_test_path = test_path + "y_test.txt"

        if self.split == "train":
            self.data = self.load_X(X_train_signals_paths)
            self.targets = self.load_y(y_train_path)
            assert len(self) == self.n_train
        elif self.split == "test":
            self.data = self.load_X(X_test_signals_paths)
            self.targets = self.load_y(y_test_path)
            assert len(self) == self.n_total - self.n_train

    def load_X(self, X_signals_paths):

        X_signals = []

        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, "r")
            # Read dataset from disk, dealing with text files' syntax
            X_signals.append(
                [
                    np.array(serie, dtype=np.float32)
                    for serie in [row.replace("  ", " ").strip().split(" ") for row in file]
                ]
            )
            file.close()

        return np.transpose(np.array(X_signals), (1, 2, 0))

    def load_y(self, y_path):
        file = open(y_path, "r")
        # Read dataset from disk, dealing with text file's syntax
        y_ = np.array(
            [elem for elem in [row.replace("  ", " ").strip().split(" ") for row in file]],
            dtype=np.int32,
        )
        file.close()

        # Substract 1 to each output class for friendly 0-based indexing
        return y_ - 1

    def download(self):
        extract_directory = os.path.abspath(self.data_path)
        if not os.path.exists(extract_directory):
            os.chdir(self.root)
            call(
                'wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip"',
                shell=True,
            )
            call('unzip -nq "UCI HAR Dataset.zip"', shell=True)
            os.remove("UCI HAR Dataset.zip")
            os.chdir("..")
