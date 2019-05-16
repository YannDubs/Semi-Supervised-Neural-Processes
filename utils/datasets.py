import os
import logging


import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

DIR = os.path.abspath(os.path.dirname(__file__))

DATASETS_DICT = {"cifar": "CIFAR10",
                 "svhn": "SVHN"}
DATASETS = list(DATASETS_DICT.keys())
UNLABELLED_CLASS = -1


def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_train_dev_test_ssl(dataset, n_labels,
                           root=None,
                           dev_size=0.1,
                           seed=123,
                           **kwargs):
    """Return the training, validation and test dataloaders

    Parameters
    ----------
    dataset : {"cifar", "svhn"}
        Name of the dataset to load

    root : str
        Path to the dataset root. If `None` uses the default one.

    dev_size : float or int
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the dev split. If int, represents the absolute
        number of dev samples.

    seed : int
        Random seed.

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    Dataset = get_dataset(dataset)

    train = Dataset(split="train") if root is None else Dataset(split="train", root=root)
    test = Dataset(split="test") if root is None else Dataset(split="test", root=root)

    train, dev = train_dev_split(train, dev_size=dev_size, seed=seed, is_stratify=True)
    make_ssl_dataset_(train, n_labels, seed=seed, is_stratify=True)

    return train, dev, test


# DATASETS
class CIFAR10(datasets.CIFAR10):
    """CIFAR10 wrapper. Docs: `datasets.CIFAR10.`

    Notes
    -----
    - Transformations (and their order) follow [1].

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test'}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.CIFAR10`.

    References
    ----------
    [1] Oliver, A., Odena, A., Raffel, C. A., Cubuk, E. D., & Goodfellow, I.
        (2018). Realistic evaluation of deep semi-supervised learning algorithms.
        In Advances in Neural Information Processing Systems (pp. 3235-3246).
    """
    img_size = (3, 32, 32)

    def __init__(self,
                 root=os.path.join(DIR, '../data/CIFAR10'),
                 split="train",
                 logger=logging.getLogger(__name__),
                 **kwargs):

        if split == "train":
            self.zca_params = dict()
            transforms_list = [transforms.Lambda(lambda x: global_contrast_norm(x)),
                               transforms.Lambda(lambda x: zca_transform(x, self.zca_params)),
                               transforms.Lambda(lambda x: horizontal_flip(x)),
                               transforms.Lambda(lambda x: random_translation(x, 2)),
                               transforms.ToTensor(),
                               # adding random noise of std 0.15
                               transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.15)]
        elif split == "test":
            self.zca_params = dict()
            transforms_list = [transforms.Lambda(lambda x: global_contrast_norm(x)),
                               transforms.Lambda(lambda x: zca_transform(x, self.zca_params)),
                               transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(root,
                         train=split == "train",
                         download=True,
                         transform=transforms.Compose(transforms_list),
                         **kwargs)

        # ZCA depends on statistics of whole data!
        # has to update because the transforms used self.zca_params as input
        # => actually the line below is a trick to change the inputs without changing
        # the functions. This works because dictionary are mutable and python uses
        # object reference
        data_zca_param = None
        if split == "train":
            if logger is not None:
                logger.info("Precomputing values for ZCA preprocesssing ...")
            data_zca_param = global_contrast_norm(self.data)

        self.zca_params.update(get_zca_params(data_zca_param, root))


class SVHN(datasets.SVHN):
    """SVHN wrapper. Docs: `datasets.SVHN.`

    Notes
    -----
    - Transformations (and their order) follow [1].

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test', "extra"}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.CIFAR10`.

    References
    ----------
    [1] Oliver, A., Odena, A., Raffel, C. A., Cubuk, E. D., & Goodfellow, I.
        (2018). Realistic evaluation of deep semi-supervised learning algorithms.
        In Advances in Neural Information Processing Systems (pp. 3235-3246).
    """
    img_size = (3, 32, 32)

    def __init__(self,
                 root=os.path.join(DIR, '../data/SVHN'),
                 split="train",
                 logger=logging.getLogger(__name__),
                 **kwargs):

        if split == "train":
            transforms_list = [transforms.ToPILImage(),
                               # max 2 pixels translation
                               transforms.RandomAffine(0,
                                                       (2 / self.img_size[1], 2 / self.img_size[2])),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x * + torch.randn_like(x) * 0.15)]
        elif split == "test":
            transforms_list = [transforms.ToTensor(),
                               transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.15)]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(root,
                         train=split == "train",
                         download=True,
                         transform=transforms.Compose(transforms_list),
                         **kwargs)


# HELPERS
def make_ssl_dataset_(supervised, n_labels,
                      unlabeled_class=UNLABELLED_CLASS, seed=123, is_stratify=True):
    """Take a supervised dataset and turn it into an unsupervised one (inplace),
    by giving a special unlabeled class as target."""
    n_all = len(supervised)
    if hasattr(supervised, "idx_mapping"):  # if splitted dataset
        idcs_all = supervised.idx_mapping
    else:
        idcs_all = list(range(n_all))

    stratify = [supervised.targets[i] for i in idcs_all] if is_stratify else None
    idcs_unlabel, indcs_labels = train_test_split(idcs_all,
                                                  stratify=stratify,
                                                  test_size=n_labels,
                                                  random_state=seed)

    supervised.n_labels = len(indcs_labels)
    for i in idcs_unlabel:
        supervised.targets[i] = unlabeled_class


class _SplitDataset(Dataset):
    """Helper to split train dataset into train and dev dataset.

    Credits: https://gist.github.com/Fuchai/12f2321e6c8fa53058f5eb23aeddb6ab
    """

    def __init__(self, to_split, length, idx_mapping):
        self.idx_mapping = idx_mapping
        self.length = length
        self.to_split = to_split

    def __getitem__(self, index):
        return self.to_split[self.idx_mapping[index]]

    def __len__(self):
        return self.length

    @property
    def targets(self):
        return self.to_split.targets


def train_dev_split(to_split, dev_size=0.1, seed=123, is_stratify=True):
    """Split a training dataset into a training and validation one.

    Parameters
    ----------
    dev_size : float or int
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the dev split. If int, represents the absolute
        number of dev samples.

    seed : int
        Random seed.

    is_stratify : bool
        Whether to stratify splits based on class label.
    """
    n_all = len(to_split)
    idcs_all = list(range(n_all))
    stratify = to_split.targets if is_stratify else None
    idcs_train, indcs_val = train_test_split(idcs_all,
                                             stratify=stratify,
                                             test_size=dev_size,
                                             random_state=seed)

    n_val = len(indcs_val)
    train = _SplitDataset(to_split, n_all - n_val, idcs_train)
    valid = _SplitDataset(to_split, n_val, indcs_val)

    return train, valid


# TRANSFORMATIONS
# all the following functions have been modified from
# `https://github.com/brain-research/realistic-ssl-evaluation/`
# for reproducability and also because tochvision buil-in transformation would
# require normalization -> to PIL -> flip, but cannot PIL normalized images
# as these are floats but shouldn't be multiplied by 255

def global_contrast_norm(images, multiplier=55, eps=1e-10):
    """Performs global contrast normalization on a array or tensor of images.

    Notes
    -----
    - Compared to the notation in the the `Deep Learning` book (p 445), `lambda=0`
        `s=multiplier`.

    Parameters
    ----------
    images: np.ndarray or torch.tensor
        Numpy array representing the original images. Shape (N, ...).

    multiplier: float, optional
        Governs severity of the adjustment.

    eps: float, optional
        Small constant to avoid divide by zero
    """
    shape = images.shape
    images = images.reshape(shape[0], -1)
    images = images.astype(float)
    # Subtract the mean of image
    images -= images.mean(axis=1, keepdims=True)
    # Divide out the norm of each image
    per_image_norm = np.linalg.norm(images, axis=1, keepdims=True)
    # Avoid divide-by-zero
    per_image_norm[per_image_norm < eps] = 1
    images = multiplier * images / per_image_norm
    return images.reshape(*shape)


def get_zca_params(images, root_path, identity_scale=0.1, eps=1e-10):
    """Creates function performing ZCA normalization on a numpy array of images.

    Parameters
    ----------
    images : np.ndarray
        Numpy array representing the original images. Shape (N, ...).

    root_path : str
        Path to save the ZCA params to.

    identity_scale : float, optional
        Scalar multiplier for identity in SVD

    eps : float, optional
        Small constant to avoid divide by zero
    """
    mean_path = os.path.join(root_path, "zca_mean.npy")
    decomp_path = os.path.join(root_path, "zca_decomp.npy")

    if os.path.exists(mean_path) and os.path.exists(decomp_path):
        image_mean = np.load(mean_path)
        zca_decomp = np.load(decomp_path)

    else:
        shape = images.shape
        images = images.reshape(shape[0], -1)

        image_covariance = np.cov(images, rowvar=False)
        U, S, _ = np.linalg.svd(
            image_covariance + identity_scale * np.eye(*image_covariance.shape)
        )
        zca_decomp = np.dot(U, np.dot(np.diag(1. / np.sqrt(S + eps)), U.T))
        image_mean = images.mean(axis=0)

        if not os.path.exists(root_path):
            os.makedirs(root_path)
        np.save(mean_path, image_mean)
        np.save(decomp_path, zca_decomp)

    return dict(zca_mean=image_mean,
                zca_decomp=zca_decomp)


def zca_transform(images, zca_params):
    """
    Apply a ZCA transformation on an array.
    """
    shape = images.shape
    images = images.reshape(shape[0], -1)
    return np.dot(images - zca_params["zca_mean"], zca_params["zca_decomp"]
                  ).reshape(*shape)


def horizontal_flip(images):
    """Random horizontal flips with proba 0.5 on np.ndarray (B H W C)."""
    idx_w = 2
    batch_size = images.shape[0]
    flipped_images = np.flip(images, idx_w)
    is_flips = np.random.randint(0, 2, size=[batch_size, 1, 1, 1])
    return is_flips * flipped_images + (1 - is_flips) * flipped_images


def random_translation(images, max_pix):
    """
    Random translations of 0 to max_pix given np.ndarray (B H W C).
    """
    idx_h, idx_w = 1, 2
    batch_size = images.shape[0]

    images = np.pad(images, [[0, 0], [max_pix, max_pix], [max_pix, max_pix], [0, 0]],
                    mode="reflect")

    shifts = np.random.randint(-max_pix, max_pix + 1, size=[batch_size, 2])  # H and W

    # -1 because removes the batch dim
    processed_data = np.stack([np.roll(images[i, ...], shift, (idx_h - 1, idx_w - 1))
                               for i, shift in enumerate(shifts)], axis=0)

    cropped_data = processed_data[:, max_pix:-max_pix, max_pix:-max_pix, :]
    return cropped_data
