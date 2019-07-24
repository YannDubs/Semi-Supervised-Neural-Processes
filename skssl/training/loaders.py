import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from sklearn.utils import resample, shuffle

from skorch.utils import to_numpy, to_tensor


class SSLBatchSampler(Sampler):
    """
    Samples a batch for semi supervised learning such that half of the batch contains
    unlabelled data while the rest contains stratified samples of the labelled data.

    Parameters
    ----------
    targets: array like
        Array like of targets. -1 should be for unlabelled.

    num_samples: int
        Batch size.
    """

    def __init__(self, targets, batch_size):
        self.targets = to_numpy(targets)
        self.batch_size_lab = batch_size // 2
        self.batch_size_unlab = batch_size - self.batch_size_lab

    def __iter__(self):
        new_idcs, targets = shuffle(np.arange(len(self.targets)), self.targets)

        idcs_unlab = (targets == -1).nonzero()[0]
        idcs_lab = (targets != -1).nonzero()[0]
        idcs_lab = resample(idcs_lab,
                            n_samples=len(idcs_unlab),
                            stratify=targets[idcs_lab],
                            replace=True)

        for i in range(0, len(idcs_unlab), self.batch_size_unlab):
            yield np.hstack([new_idcs[idcs_unlab[i:i + self.batch_size_unlab]],
                             new_idcs[idcs_lab[i:i + self.batch_size_lab]]])

    def __len__(self):
        return len(self.targets)


def get_ssl_iterator(dataset, batch_size=128, **kwargs):
    """Initializes the SSl iterator. Can be given as NeuralNet.iterator_*."""
    try:
        y = dataset.targets
    except AttributeError:
        y = dataset.y
    return DataLoader(dataset,
                      batch_sampler=SSLBatchSampler(y, batch_size=batch_size),
                      **kwargs)


class SupervisedSampler(Sampler):
    """
    Samples a batch for supervised learning such that batch contains only labelled
    data. Stratify resampling such that the number of epochs does not have to change.

    Parameters
    ----------
    targets: array like
        Array like of targets. -1 should be for unlabelled.

    num_samples: int
        Batch size.
    """

    def __init__(self, targets):
        targets = to_numpy(targets)
        self.targets_idcs = (targets != -1).nonzero()[0]
        self.targets = targets[self.targets_idcs]

    def __iter__(self):
        idcs = shuffle(self.targets_idcs)
        return iter(idcs)

    def __len__(self):
        return len(self.targets)


def get_supervised_iterator(dataset, **kwargs):
    """Initializes the supervised iterator (disregard -1). Can be given as NeuralNet.iterator_*."""
    try:
        y = dataset.targets
    except AttributeError:
        y = dataset.y

    kwargs = {k: v for k, v in kwargs.items() if k != "shuffle"}
    # remove shuffle because you have a sampler => mutually exclusive

    return DataLoader(dataset,
                      sampler=SupervisedSampler(y),
                      **kwargs)


def batch_transforms_collate(batch_transforms=[]):
    """Return a collate function that applies the given transforms."""
    def transformed_collate(batch):
        collated = torch.utils.data.dataloader.default_collate(batch)
        for transform in batch_transforms:
            collated = transform(collated)
        return collated
    return transformed_collate
