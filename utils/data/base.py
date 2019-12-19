import copy
import logging
import os

import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from wildml.utils.helpers import tmp_seed
from wildml.utils.datasplit import RandomMasker

DIR = os.path.abspath(os.path.dirname(__file__))

UNLABELLED_CLASS = -1


class BaseDataset:
    """BaseDataset that should be inherited by the format-specific ones.
    
    Parameters
    ----------
    root : str, optional
        Root to the data directory.

    logger : logging.Logger, optional
        Logger

    is_return_index : bool, optional
        Whether to return the index in addition to the labels.

    seed : int, optional
        Random seed.
    """

    unlabelled_class = UNLABELLED_CLASS

    def __init__(
        self,
        root=os.path.join(DIR, "../../../../data/"),
        logger=logging.getLogger(__name__),
        is_return_index=False,
        seed=123,
    ):
        self.seed = seed
        self.logger = logger
        self.root = root
        self.is_return_index = is_return_index

    def rm_all_transformations_(self):
        """Completely remove transformation."""
        pass

    def make_test_(self):
        """Make the data a test set."""
        pass

    def train_test_split(self, test_size=0.1, is_stratify=True):
        """Split the dataset into train and test (without data augmentation).

        Parameters
        ----------
        test_size : float or int, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of
            the dataset to include in the test split. If int, represents the absolute
            size of the test dataset. 

        is_stratify : bool, optional
            Whether to stratify splits based on class label.

        Returns
        ------- 
        train : BaseDataset
            Train dataset containing the complement of `test_size` examples.

        test : BaseDataset
            Test dataset containing `test_size` examples.
        """
        idcs_all = list(range(len(self)))
        stratify = self.targets if is_stratify else None
        idcs_train, indcs_test = train_test_split(
            idcs_all, stratify=stratify, test_size=test_size, random_state=self.seed
        )

        train = self.clone()
        train.keep_indcs_(idcs_train)

        test = self.clone()
        test.keep_indcs_(indcs_test)
        test.make_test_()

        return train, test

    def drop_labels_(self, drop_size, is_stratify=True):
        """Drop part of the labels to make the dataset semisupervised.
        
        Parameters
        ----------
        drop_size : float or int or tuple, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of the labels to 
            drop. If int, represents the number of labels to drop. 0 means keep all.

        is_stratify : bool, optional
            Whether to stratify splits based on class label.
        """
        if drop_size == 0:
            return

        self.logger.info(f"Dropping {drop_size} labels...")

        idcs_all = list(range(len(self)))
        stratify = self.targets if is_stratify else None
        idcs_label, idcs_unlabel = train_test_split(
            idcs_all, stratify=stratify, test_size=drop_size, random_state=self.seed
        )

        self.targets[idcs_unlabel] = self.unlabelled_class

    def balance_labels_(self):
        """
        Balances the number of labelled and unlabbeld data by updasmpling labeled. Only works if
        number of labelled data is smaller than unlabelled.
        """
        self.logger.info(f"Balancing the semi-supervised labels...")

        idcs_unlab = [i for i, t in enumerate(self.targets) if t == UNLABELLED_CLASS]
        idcs_lab = [i for i, t in enumerate(self.targets) if t != UNLABELLED_CLASS]

        assert len(idcs_unlab) > len(idcs_lab)

        resampled_idcs_lab = resample(
            idcs_lab,
            replace=True,
            n_samples=len(idcs_unlab),
            stratify=self.targets[idcs_lab],
            random_state=self.seed,
        )

        self.keep_indcs_(idcs_unlab + resampled_idcs_lab)

    def drop_unlabelled_(self):
        """Drop all the unlabelled examples."""
        self.logger.info(f"Drop all the unlabelled examples...")

        idcs_lab = [i for i, t in enumerate(self.targets) if t != UNLABELLED_CLASS]
        self.keep_indcs_(idcs_lab)

    def clone(self):
        """Returns a deepcopy of the daatset."""
        return copy.deepcopy(self)

    def keep_indcs_(self, indcs):
        """Keep the given indices.
        
        Parameters
        ----------
        indcs : array-like int
            Indices to keep. If the multiplicity of the indices is larger than 1 then will duplicate
            the data.
        """
        self.data = self.data[indcs]
        self.targets = self.targets[indcs]

    def drop_features_(self, drop_size):
        """Drop part of the features (e.g. pixels in images).

        Parameters
        ----------
        drop_size : float or int or tuple, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to 
            drop. If int, represents the number of datapoints to drop. If tuple, same as before 
            but give bounds (min and max). 1 means drop all.
        """
        if drop_size == 0:
            return

        raise NotImplementedError("drop_features_ not implemented for current dataset")

    def add_index(self, y, index):
        """Append the index to the targets (if needed)."""
        if self.is_return_index:
            try:
                y = tuple(y) + (index,)
            except TypeError:
                y = [y, index]
        return y
