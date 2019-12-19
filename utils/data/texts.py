import numpy as np
import torch

from torchtext.datasets import text_classification

from wildml.utils.helpers import cont_tuple_to_tuple_cont

from .base import BaseDataset
from .helpers import get_masks_drop_features

DATASETS_DICT = {"ag_news": "AGNews", "yahoo": "YahooAnswers", "dbpedia": "DBpedia"}
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


class TextDatset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_drop_features = False

    def drop_features_(self, drop_size):
        """Drop part of the features (words in sentences) by seeting them to the <unk> token.

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
            lambda i: (len(self.data[i]),),
            len(self),
            seed=self.seed,
            n_batch=1,  # has to use 1 per batch because they are all of different number of words
        )

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

    def __getitem__(self, idx):
        X = self.data[idx]

        if self.is_drop_features:
            X[self.to_drop[idx]] = self.vocab.unk_index

        return X, self.targets[idx]


class TorchTextClassification(TextDatset):
    """Base class for any torchtex.text_classification.

    Notes
    -----
    - tokenization is `basic_english` which lowercases and adds spaces between punctuation.
    
    Parameters
    ----------
    name : str
        Name of the torchtext classification dataset.

    ngrams : int, optional
        Number of n-grams.

    split : {'train', 'test', ...}, optional
        According dataset is selected.

    kwargs :
        Additional arguments to TextDatset.
    """

    def __init__(self, name, ngrams=2, split="train", **kwargs):
        super().__init__(**kwargs)
        out = text_classification.DATASETS[name](self.root, ngrams=ngrams)

        if split == "train":
            data = out[0]
        if split == " test":
            data = out[1]

        self.vocab = data._vocab
        self.targets, self.data = cont_tuple_to_tuple_cont(data._data)
        self.data = self.data
        self.targets = torch.tensor(self.targets)


class AGNews(TorchTextClassification):
    """AGNews dataset.

    Parameters
    ----------
    kwargs : 
        Additional parameters to TorchTextClassification.

    Examples
    --------
    >>> data = AGNews(split="train")
    >>> len(data)
    120000
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
    216000
    >>> data.drop_unlabelled_()
    >>> len(data)
    108000

    >>> data.drop_features_(0.7)
    >>> round((data[0][0] == data.vocab.unk_index).float().mean().item(), 1)
    0.7
    >>> data[0][0] 
    tensor([     0,      0,      0,   4089,    191,      0,      0,      0,      0,
             82030,      0,      0,      0,      0,    615,      0,   5724,     76,
                 0,    376,      0,      0,      3,      0,      0,  21377,      0,
                 0,      0,      0,      0,      0,    302,      2,      0,      0,
            231620,      0,      0,      0,      0,      0,      0,      0,      0,
                 0,    120,  13390,      0, 210030,      0,      0,      0,      0,
                 0,    207,      0, 974824,      0, 480540,      0,   1737,      0,
                 0, 265226,      0,   2862])
    """

    n_train = 120000
    n_classes = 4

    def __init__(self, **kwargs):
        super().__init__("AG_NEWS", **kwargs)


class YahooAnswers(TorchTextClassification):
    """YahooAnswers dataset.

    Parameters
    ----------
    kwargs : 
        Additional parameters to TorchTextClassification.

    Examples
    --------
    See AGNews
    """

    n_classes = 4

    def __init__(self, **kwargs):
        super().__init__("YahooAnswers", **kwargs)


class DBpedia(TorchTextClassification):
    """DBpedia dataset.

    Parameters
    ----------
    kwargs : 
        Additional parameters to TorchTextClassification.

    Examples
    --------
    See AGNews
    """

    n_train = 560000
    n_classes = 14

    def __init__(self, **kwargs):
        super().__init__("DBpedia", **kwargs)

