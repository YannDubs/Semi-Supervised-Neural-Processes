import numpy as np
from sklearn.model_selection import PredefinedSplit


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
