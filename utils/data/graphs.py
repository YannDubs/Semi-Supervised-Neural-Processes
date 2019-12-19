import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric import datasets as geom_datasets

from .base import BaseDataset
from .helpers import get_masks_drop_features

DATASETS_DICT = {"enzymes": "Enzymes", "proteins": "Proteins", "synthie": "Synthie"}
DATASETS = list(DATASETS_DICT.keys())

# HELPERS
def get_Dataset(dataset):
    """Return the correct uninstantiated datasets."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return partial(graph_splitter, Dataset=eval(DATASETS_DICT[dataset]))
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def graph_splitter(Dataset, *args, split="train", **kwargs):
    """Return the correspoding splitted `GraphsDataset`."""
    dataset = Dataset(*args, **kwargs)

    train, test = dataset.train_test_split(test_size=0.1, is_stratify=True)

    if split == "train":
        dataset = train
    elif split == "test":
        dataset = test
    else:
        raise ValueError(f"Unkown split={split}")

    return dataset


class GraphDataset(BaseDataset):
    """Graph dataset wrapper that adds nice functionalities.
    
    Parameters
    ----------
    transformer : sklearn.TransformerMixin, optional
        Transformation to apply to all the node values at once.
    """

    def __init__(self, *args, transformer=StandardScaler(), **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = transformer
        self.is_drop_features = False  # make sure that drops only once for now

    def transform_(self):
        if self.transformer is not None:
            #! not great because when standardizing, you do it on both train and test :/
            self.data = torch.from_numpy(self.transformer.fit_transform(self.data.numpy()))

    def __getitem__(self, index):

        data = self.graphs[index]

        if self.is_drop_features:
            data.x[self.to_drop[index], :] = float("nan")

        if data.edge_attr is None:
            # if no weighting of edges use 1 everywhere for adjacency matrix
            edge_attr = torch.ones_like(data.edge_index[0], dtype=torch.float)
        else:
            edge_attr = data.edge_attr

        X = dict(
            x=data.x,
            adj=torch.sparse.FloatTensor(
                data.edge_index,
                edge_attr,
                size=[data.num_nodes, data.num_nodes],
                device=data.x.device,
            ),
        )

        target = data.y.item()
        target = self.add_index(target, index)

        return X, target

    def keep_indcs_(self, indcs):
        """Keep only the given indices."""
        self.graphs = self.graphs.__indexing__(indcs)

    def drop_features_(self, drop_size):
        """Drop part of the features (node values in graphs).

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
            drop_size,
            lambda i: (self.graphs[i].num_nodes,),
            len(self),
            seed=self.seed,
            n_batch=1,  # has to use 1 per batch because they are all of different number of nodes
        )

    @property
    def data(self):
        return self.graphs.data.x

    @data.setter
    def data(self, values):
        self.graphs.data.x = values

    @property
    def targets(self):
        return self.graphs.data.y

    @targets.setter
    def targets(self, values):
        self.graphs.data.y = values

    def __len__(self):
        return len(self.graphs)


# PYTORCH GEOMETRIC DATASET
class TUDataset(GraphDataset):
    """Base TUDataset dataset.`

    Parameters
    ----------
    name : str
        Name of the dataset.
        
    root : str, optional
        Path to the dataset root.

    kwargs:
        Additional arguments to `datasets.TUDataset` and `GraphDataset`.
    """

    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.graphs = geom_datasets.TUDataset(
            root=f"{self.root}/{name}", name=name, use_node_attr=True
        )
        self.transform_()


class Enzymes(TUDataset):
    """Enzymes dataset.

    Parameters
    ----------
    **kwargs:
        Additional arguments to TUDataset.
    
    Examples
    --------
    >>> data = Enzymes()
    >>> len(data)
    600
    
    # data.data not the same length because all nodes seen together
    >>> len(data) == len(data.data) == len(data.targets) 
    False
    >>> [type(i) for i in data[0]]
    [<class 'dict'>, <class 'int'>]
    >>> list(data[0][0].keys())
    ['x', 'adj']

    >>> train, valid = data.train_test_split(test_size=100, is_stratify=True)
    >>> len(valid)
    100

    >>> data.drop_labels_(0.9)
    >>> round(len([t for t in data.targets if t == -1]) / len(data), 1)
    0.9

    >>> data.balance_labels_()
    >>> len(data)
    1080
    >>> data.drop_unlabelled_()
    >>> len(data)
    540

    >>> data.drop_features_(0.7)
    >>> round((torch.isnan(data[0][0]["x"])).float().mean().item(), 1)
    0.7
    >>> data[0][0]["x"]
    tensor([[    nan,     nan,     nan,  ...,     nan,     nan,     nan],
            [    nan,     nan,     nan,  ...,     nan,     nan,     nan],
            [ 0.1920, -0.1453,  0.7947,  ...,  1.0346, -0.9873, -0.1548],
            ...,
            [-0.5522, -0.3998, -0.2392,  ..., -0.9665,  1.0129, -0.1548],
            [    nan,     nan,     nan,  ...,     nan,     nan,     nan],
            [    nan,     nan,     nan,  ...,     nan,     nan,     nan]])
    """

    n_train = 540
    n_classes = 6

    def __init__(self, *args, **kwargs):
        super().__init__("ENZYMES", *args, **kwargs)


class Proteins(TUDataset):
    """PROTEINS_full dataset.
    
    Parameters
    ----------
    **kwargs:
        Additional arguments to TUDataset.
    
    Examples
    --------
    See Enzymes.
    """

    n_train = 1001
    n_classes = 2

    def __init__(self, *args, **kwargs):
        super().__init__("PROTEINS_full", *args, **kwargs)


class Synthie(TUDataset):
    """Synthie dataset.

    Parameters
    ----------
    **kwargs:
        Additional arguments to TUDataset.
    
    Examples
    --------
    See Enzymes.
    """

    n_train = 360
    n_classes = 4

    def __init__(self, *args, **kwargs):
        super().__init__("Synthie", *args, **kwargs)
