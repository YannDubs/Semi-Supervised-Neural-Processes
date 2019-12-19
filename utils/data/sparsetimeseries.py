from functools import partial
import os

import numpy as np
import pandas as pd
import h5py
import torch

from .base import BaseDataset

DATASETS_DICT = {"physionet": "Physionet2012"}
DATASETS = list(DATASETS_DICT.keys())


def get_Dataset(dataset):
    """Return the correct uninstantiated datasets."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


class SparseMultiTimeSeriesDataset(BaseDataset):
    """Pytorch wrapper for sparse multidimensional time series.

    Notes
    -----
    - The hdf5 dataset should be a dataset with groups "train", "test", "target", 
    (optional) "metadata", (optional) "mapper". Each group should have any number of dataset 
    corresponding to different multi-dimensional times series indexed by their name. Each multi 
    dimensional time series should be represented as 2 column pandas dataframe. The dimension 
    (int) called "Parameter", and the value (float) called "Value" and indexed by time 
    (float in [-1,1]) called "Time". Target are assumed to be in order.
    - here we don't assume that the time series have values seen in block : can have missing values
    but different on each dimension.
    - No `drop_features_` because features are already droped before hand.

    Parameters
    ----------
    path : str
        Path to the hdf5 dataset from the root.
    
    split : {"train", "test", "both"}, optional
        Which set to load.
    
    name_to_id : callable, optional
        Function mapping the time series names to their ids. Typically used because ID is an integer
        while name is a string.

    y_idx : int or string
        Selects only one dimension fo the multidimensional time series. Of string, need "mapper"
        to be given.
    """

    def __init__(self, path, split="train", name_to_id=lambda s: int(s[1:]), y_idx=None):
        super().__init__()

        self.path = os.path.join(self.root, path)
        self.split = split
        self.name_to_id = name_to_id
        self.y_idx = y_idx

        if self.split == "train":
            self.data = self.load_keys("train")
        elif self.split == "test":
            self.data = self.load_keys("test")
        else:
            self.data = self.load_keys("train") + self.load_keys("test")

        # always read in whole targets and metadata because smallish
        self.targets = pd.read_hdf(self.path, "target")
        self.targets = torch.from_numpy(
            self.targets.loc[[self.name_to_id(self._key_to_name(k)) for k in self.data]].values
        ).float()

        try:
            self.metadata = pd.read_hdf(self.path, "metadata")
            self.metadata = torch.from_numpy(
                self.metadata.loc[
                    [self.name_to_id(self._key_to_name(k)) for k in self.data]
                ].values
            ).float()
        except KeyError:
            self.metadata = None

        try:
            self.mapper = pd.read_hdf(self.path, "mapper")
        except KeyError:
            self.metadata = None

        if isinstance(y_idx, str):
            self.y_idx = dict(mapper)[self.y_idx]

    def load_keys(self, group):
        with h5py.File(self.path, mode="r") as f:
            keys = ["{}/{}".format(group, k) for k in list(f[group].keys())]
        return keys

    def _key_to_name(self, k):
        return k.rsplit("/", 1)[-1]

    def keep_indcs_(self, indcs):
        """Keep the given indices.
        
        Parameters
        ----------
        indcs : array-like int
            Indices to keep. If the multiplicity of the indices is larger than 1 then will duplicate
            the data.
        """
        self.data = [self.data[i] for i in indcs]
        self.targets = self.targets[indcs]

    def __len__(self):
        return len(self.data)

    def _format_ts(self, ts):
        # tensor with 2 columns : time and (int) name of multi each time series
        X = torch.from_numpy(ts.reset_index()[["Parameter", "Time"]].values).float()
        # actual time series
        y = torch.from_numpy(ts.Value.values).unsqueeze(1).float()
        return X, y

    def __getitem__(self, idx):
        """
        Returns
        -------
        data : dictionary
            - "X" : features for the inputs. Torch tensor of shape (*,2). The first column is the 
                    mutlitimeseries / channel idx the second column is the time .
            - "y" : actual values of the time series.

        target : int or list-like
        """
        key = self.data[idx]
        ts = pd.read_hdf(self.path, key)

        X, y = self._format_ts(ts)

        if self.y_idx is not None:
            # select only one channel / time series (Not multi dim)
            channels = X[..., 0]
            mask = channels == self.y_idx
            assert mask.sum() != 0, "When choses on time series, it should be non empty"

            X = X[mask]
            y = y[mask]

        data = dict(X=X, y=y)

        if self.metadata is not None:
            data["metadata"] = self.metadata[idx]

        target = self.targets[idx]
        target = self.add_index(target, idx)

        return data, target


class Physionet2012(SparseMultiTimeSeriesDataset):
    """Physionet 2012 challenge with multi(4)-target classifier as discussed in [1]. The first
    target is the one of mortality whcih is the other task in [1].

    Parameters
    ----------
    kwargs :
        Additional arguments to `SparseMultiTimeSeriesDataset`.

    References
    ----------
    [1] https://www.nature.com/articles/s41598-018-24271-9#MOESM1

    Examples
    --------
    >>> data = Physionet2012(split="train") 
    >>> len(data)
    ???
    >>> [type(i) for i in data[0]]
    [<class 'dict'>, <class 'int'>]

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
    """

    n_classes = 4  # but MUTLI TARGET
    unlabelled_class = [-1, -1, -1, -1]

    def __init__(self, **kwargs):
        super().__init__("Physionet2012/data.hdf5", **kwargs)

