from functools import partial

import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset
import torch

from econvcnp.utils.helpers import indep_shuffle_, ratio_to_int

PATHS = {"physionet2012": "data/Physionet2012/data.hdf5"}
DATASETS = list(PATHS.keys())


def get_timeseries_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()

    if dataset == "har":
        return HARDataset

    try:
        return partial(SparseMultiTimeSeriesDataset, path=PATHS[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


class SparseMultiTimeSeriesDataset(Dataset):
    """PYtorch wrapper of any sparse multidimensional time series.

    Parameters
    ----------
    path : str
        Path to the datasat. This should be a hdf5 dataset whith groups "train", "test", "target", (optional)
        "metadata", (optional) "mapper". Each group should have any number of dataset corresponding to different
        multi-dimensional times series indexed by their name. Each multi dimensional time series should be
        represented as 2 column pandas dataframe the dimension (int) called "Parameter", and the value (float)
        called "Value" and indexed by time (float in [-1,1]) called "Time". Target are assumed to be in order.

    split : {"train", "test", "both"}, optional
        Which set to load.

    name_to_id : callable, optional
        Function mapping the time series names to their ids. Typically used because ID is a integer
        while name is a string.

    get_cntxt_trgt : callable, optional
        Function that split the input into context and target points.
        `X_cntxt, Y_cntxt, X_trgt, Y_trgt = self.get_cntxt_trgt(X, y)`.


    """

    def __init__(self, path,
                 split="train",
                 name_to_id=lambda s: int(s[1:]),
                 get_cntxt_trgt=None,
                 y_idx=None,
                 is_show_dense=False):

        self.path = path
        self.split = split
        self.name_to_id = name_to_id
        self.get_cntxt_trgt = get_cntxt_trgt
        self.y_idx = y_idx
        self.is_show_dense = is_show_dense
        if is_show_dense:
            assert y_idx is not None

        if self.split == "train":
            self.keys = self.load_keys("train")
        elif self.split == "test":
            self.keys = self.load_keys("test")
        else:
            self.keys = self.load_keys("train") + self.load_keys("test")

        # always read in whole targets and metadata because smallish
        self.targets = pd.read_hdf(self.path, "target")
        self.targets = torch.from_numpy(self.targets.loc[[self.name_to_id(self._key_to_name(k))
                                                          for k in self.keys]].values).float()

        try:
            self.metadata = pd.read_hdf(self.path, "metadata")
            self.metadata = torch.from_numpy(self.metadata.loc[[self.name_to_id(self._key_to_name(k))
                                                                for k in self.keys]].values).float()
        except KeyError:
            self.metadata = None

        try:
            self.mapper = pd.read_hdf(self.path, "mapper")
        except KeyError:
            self.metadata = None

    def load_keys(self, group):
        with h5py.File(self.path, mode="r") as f:
            keys = ["{}/{}".format(group, k) for k in list(f[group].keys())]
        return keys

    def _key_to_name(self, k):
        return k.rsplit('/', 1)[-1]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        ts = pd.read_hdf(self.path, k)
        data = {}

        X, y = self._format_ts(ts)

        if self.y_idx is not None:
            # select only one channel
            channels = X[..., 0]
            mask = channels == self.y_idx
            if mask.sum() == 0:
                return self[(idx + 1) % len(self)]  # return next if empty

            if self.is_show_dense:
                X = X[mask][..., 1:]  # remove channel
            y = y[mask]

        if self.get_cntxt_trgt is None:
            # shuffle (because when to mycollate will take only the first min_length)
            permuted_idcs = torch.randperm(X.size(0))
            data["X"] = X[permuted_idcs, ...]
            data["y"] = y[permuted_idcs, ...]
        else:
            X, y = X.unsqueeze(0), y.unsqueeze(0)  # add batch dim
            data["X_cntxt"], data["y_cntxt"], data["X_trgt"], data["y_trgt"] = (x.squeeze(0) for x in
                                                                                self.get_cntxt_trgt(X, y))

        if self.metadata is not None:
            data["metadata"] = self.metadata[idx]

        return data, self.targets[idx]

    def _format_ts(self, ts):
        X = torch.from_numpy(ts.reset_index()[["Parameter", "Time"]].values).float()
        y = torch.from_numpy(ts.Value.values).unsqueeze(1).float()
        return X, y


# dirty code for HAR
# credit https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition


class HARDataset(Dataset):
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]

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
        "total_acc_z_"
    ]

    n_train = 7352
    n_total = 10299

    def __init__(self, path="data/", split="train", data_perc=1, is_fill_mean=False,
                 transformed_file=None):

        data_path = path + "UCI HAR Dataset/"
        train_path = data_path + self.TRAIN
        test_path = data_path + self.TEST

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

        self.transformed_file = transformed_file
        self.split = split
        self.data_perc = data_perc
        self.is_fill_mean = is_fill_mean

        if self.split == "train":
            if self.transformed_file is not None:
                transformed_data = np.load(self.transformed_file, allow_pickle=False)
                self.data = transformed_data[:self.n_train]
            else:
                self.data = self.load_X(X_train_signals_paths)
            self.targets = self.load_y(y_train_path)
            assert len(self) == self.n_train
        elif self.split == "test":
            if self.transformed_file is not None:
                transformed_data = np.load(self.transformed_file, allow_pickle=False)
                self.data = transformed_data[self.n_train - self.n_total:]
            else:
                self.data = self.load_X(X_test_signals_paths)
            self.targets = self.load_y(y_test_path)
            assert len(self) == self.n_total - self.n_train
        else:
            if self.transformed_file is not None:
                transformed_data = np.load(self.transformed_file, allow_pickle=False)
                self.data = transformed_data
            else:
                self.data = np.concatenate([self.load_X(X_train_signals_paths),
                                            self.load_X(X_test_signals_paths)])

            self.targets = np.concatenate([self.load_y(y_train_path),
                                           self.load_y(y_test_path)])
            assert len(self) == self.n_total

        self.n_possible_points = self.data.shape[1]
        n_indcs = ratio_to_int(self.data_perc, self.n_possible_points)
        filename = data_path + "indcs_{}.npy".format(n_indcs)
        try:
            indcs = np.load(filename)
        except FileNotFoundError:
            len_total_data = self.n_total
            indcs = np.arange(self.n_possible_points).reshape(1, -1).repeat(len_total_data, axis=0)
            indep_shuffle_(indcs, -1)
            indcs = indcs[:, :n_indcs]
            indcs.sort()
            np.save(filename, indcs)

        if self.split == "train":
            self.indcs = indcs[:self.n_train, :]
        elif self.split == "test":
            self.indcs = indcs[self.n_train:, :]
        else:
            self.indcs = indcs

        if self.transformed_file is None:
            # standardize with the training mean and std
            self.data = (self.data - np.array([3.4212677e-09, 4.8643618e-09, 3.9563477e-09,
                                               -2.1403190e-09, 6.1939538e-09, -3.5347694e-09,
                                               -7.3419429e-08, 6.4858154e-09, -2.5424397e-08])
                         ) / np.array([0.19484635, 0.12242749, 0.10687881, 0.40681508, 0.38185433,
                                       0.25574315, 0.4141119, 0.3909954, 0.35776883])

    def load_X(self, X_signals_paths):

        X_signals = []

        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, 'r')
            # Read dataset from disk, dealing with text files' syntax
            X_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                ]]
            )
            file.close()

        return np.transpose(np.array(X_signals), (1, 2, 0))

    def load_y(self, y_path):
        file = open(y_path, 'r')
        # Read dataset from disk, dealing with text file's syntax
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]],
            dtype=np.int32
        )
        file.close()

        # Substract 1 to each output class for friendly 0-based indexing
        return y_ - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # drop whole multi dimensinla time serie
        indcs = torch.from_numpy(self.indcs[idx])

        if self.is_fill_mean:
            Y = torch.from_numpy(self.data[idx]).float()
            # select all not in indcs
            mask = torch.ones_like(Y).byte()
            mask[indcs] = 0
            # fill in with mean of Y
            Y = torch.where(mask, Y.mean(0), Y)
            X = torch.linspace(-1, 1, Y.size(0)).unsqueeze(1).float()
        else:
            Y = torch.from_numpy(self.data[idx]).float()[indcs]
            X = torch.linspace(-1, 1, self.n_possible_points).unsqueeze(1).float()[indcs]

        t = torch.from_numpy(self.targets[idx]).long().squeeze()
        return {"X": X, "y": Y}, t
