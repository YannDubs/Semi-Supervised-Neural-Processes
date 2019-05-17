import os
import logging
import pickle

from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.datasets
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.transforms.functional import to_tensor
import skorch.dataset

DIR = os.path.abspath(os.path.dirname(__file__))

DATASETS_DICT = {"cifar": "CIFAR10",
                 "svhn": "SVHN",
                 "pts_circles": None,
                 "pts_moons": None,
                 "pts_var_gaus": None,
                 "pts_cov_gaus": None,
                 "pts_iso_gaus": None,
                 }
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
        Additional arguments to `generate_train_dev_test_ssl`.
    """
    Dataset = get_dataset(dataset)

    if Dataset is None:
        # has to generate
        return generate_train_dev_test_ssl(dataset, n_labels,
                                           dev_size=dev_size,
                                           seed=seed,
                                           **kwargs)

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
            transforms_list = [transforms.Lambda(lambda x: horizontal_flip(x)),
                               transforms.Lambda(lambda x: random_translation(x, 2)),
                               transforms.ToTensor(),
                               # adding random noise of std 0.15
                               transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.15)]
        elif split == "test":
            transforms_list = [transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(root,
                         train=split == "train",
                         download=True,
                         transform=transforms.Compose(transforms_list),
                         **kwargs)

        # Precomputing to make it quicker if load multiple time
        path_precomputed_data = os.path.join(root, "clean_{}_data.npy".format(split))
        path_precomputed_target = os.path.join(root, "clean_{}_target.pkl".format(split))
        if os.path.exists(path_precomputed_data) and os.path.exists(path_precomputed_target):
            # save also targets even though not preprocesed
            # but want to be sure that correct order
            self.data = np.load(path_precomputed_data)
            with open(path_precomputed_target, 'rb') as f:
                self.targets = pickle.load(f)
        else:
            logger.warning("Precomputing preprocessed data ...")

            self.data = global_contrast_norm(self.data)
            if split == "train":
                # only recomputing if train else load precomputed
                zca_params = get_zca_params(self.data, root)
            else:
                zca_params = get_zca_params(None, root)
            self.data = zca_transform(self.data, zca_params)

            # use float32
            # usually not needed because converted to PIL but we removed
            # due to preprocessing
            self.data = self.data.astype(np.float32)  # USE FLOAT32

            np.save(path_precomputed_data, self.data)
            with open(path_precomputed_target, 'wb') as f:
                pickle.dump(self.targets, f)

    def __getitem__(self, index):
        """Changes dfault to not convert to PIL due to preprocessing"""
        img, target = self.data[index], self.targets[index]

        # removed PIL conversion due to preprocessing => bad type
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


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
            transforms_list = [transforms.Lambda(lambda x: random_translation(x, 2)),
                               # put image in [-1,1]
                               transforms.Lambda(lambda x: (to_tensor(x) - 0.5) * 2)]
        elif split == "test":
            transforms_list = [transforms.Lambda(lambda x: (to_tensor(x) - 0.5) * 2)]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(root,
                         split="train",
                         download=True,
                         transform=transforms.Compose(transforms_list),
                         **kwargs)

    @property
    def targets(self):
        # make compatible with CIFAR10 dataset
        return self.labels


def generate_train_dev_test_ssl(dataset, n_label,
                                n_unlabel=int(1e4),
                                n_test=int(1e4),
                                dev_size=0.1, seed=123, is_hard=False):
    n_dev = int(n_unlabel * dev_size)
    hard_factor = 1.5 if is_hard else 1  # multiply by 1.5 if hard

    gaus_means = np.array([[7, 9], [8., -1], [-5., -9.]])
    args = dict(pts_circles={"f": sklearn.datasets.make_circles,
                             "std": 0.09, "kwargs": dict(factor=.5)},
                pts_moons={"f": sklearn.datasets.make_moons,
                           "std": 0.13, "kwargs": {}},
                pts_iso_gaus={"f": sklearn.datasets.make_blobs,
                              "std": 1.5, "kwargs": dict(centers=gaus_means)},
                pts_cov_gaus={"f": sklearn.datasets.make_blobs,
                              "std": 1.5,
                              "kwargs": dict(centers=gaus_means)},
                pts_var_gaus={"f": sklearn.datasets.make_blobs,
                              "std": np.array([1.0, 2.5, 0.5]),
                              "kwargs": dict(centers=gaus_means)})

    spec_args = args[dataset]

    def get_noisy_kwargs(is_label=False):
        if "_gaus" in dataset:
            std_label_factor = 1 if not is_label else 0.5  # divide by 10 std
            spec_args["kwargs"]["cluster_std"] = spec_args["std"] * std_label_factor * hard_factor
        else:
            std_label_factor = 1 if not is_label else 0  # no noise
            spec_args["kwargs"]["noise"] = spec_args["std"] * std_label_factor * hard_factor
        return spec_args["kwargs"]

    X_lab, y_lab = spec_args["f"](n_samples=n_label, random_state=seed,
                                  **get_noisy_kwargs(True))
    X_unlab, y_unlab = spec_args["f"](n_samples=n_unlabel, random_state=seed,
                                      **get_noisy_kwargs())
    y_unlab[:] = -1
    X_train = np.concatenate([X_lab, X_unlab])
    y_train = np.concatenate([y_lab, y_unlab])
    X_dev, y_dev = spec_args["f"](n_samples=n_dev, random_state=seed,
                                  **get_noisy_kwargs())
    X_test, y_test = spec_args["f"](n_samples=n_test, random_state=seed,
                                    **get_noisy_kwargs())

    if dataset == "pts_cov_gaus":
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_train = np.dot(X_train, transformation)
        X_dev = np.dot(X_dev, transformation)
        X_test = np.dot(X_test, transformation)

    return (skorch.dataset.Dataset(X_train, y=y_train),
            skorch.dataset.Dataset(X_dev, y=y_dev),
            skorch.dataset.Dataset(X_test, y=y_test))


# HELPERS
def make_ssl_dataset_(supervised, n_labels,
                      unlabeled_class=UNLABELLED_CLASS, seed=123, is_stratify=True):
    """Take a supervised dataset and turn it into an unsupervised one(inplace),
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

    Credits: https: // gist.github.com / Fuchai / 12f2321e6c8fa53058f5eb23aeddb6ab
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
    - Compared to the notation in the the `Deep Learning` book(p 445), `lambda=0`
        `s = multiplier`.

    Parameters
    ----------
    images: np.ndarray or torch.tensor
        Numpy array representing the original images. Shape(B, ...).

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
    images: np.ndarray
        Numpy array representing the original images. Shape(B, ...).

    root_path: str
        Path to save the ZCA params to.

    identity_scale: float, optional
        Scalar multiplier for identity in SVD

    eps: float, optional
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
    Apply a ZCA transformation on a batch array.
    """
    shape = images.shape
    images = images.reshape(shape[0], -1)
    return np.dot(images - zca_params["zca_mean"], zca_params["zca_decomp"]
                  ).reshape(*shape)


def horizontal_flip(img):
    """Random horizontal flips with proba 0.5 on np.ndarray(H W C)."""
    idx_w = 1
    flipped_img = np.flip(img, idx_w)
    is_flips = np.random.randint(0, 2, size=[1, 1, 1]).astype(np.float32)
    return is_flips * flipped_img + (1 - is_flips) * flipped_img


def random_translation(img, max_pix):
    """
    Random translations of 0 to max_pix given np.ndarray(H W C) or PIL.Image(W H C).
    """
    is_pil = not isinstance(img, np.ndarray)
    if is_pil:
        # should also transpose but this function is equivalent for H and W
        img = np.asarray(img)
    idx_h, idx_w = 0, 1
    img = np.pad(img, [[max_pix, max_pix], [max_pix, max_pix], [0, 0]],
                 mode="reflect")
    shifts = np.random.randint(-max_pix, max_pix + 1, size=[2])  # H and W
    processed_data = np.roll(img, shifts, (idx_h, idx_w))
    cropped_data = processed_data[max_pix:-max_pix, max_pix:-max_pix, :]
    if is_pil:
        img = Image.fromarray(img)
    return cropped_data
