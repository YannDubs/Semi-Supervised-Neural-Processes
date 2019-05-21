import os

import torch
from PIL import Image
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


def precompute_batch_tranforms(X, y, basename,
                               transforms_X=[],
                               transforms_y=[],
                               logger=None):
    """Apply and store batch transforms to `X` and `y` or load precomputed
    ones.

    Parameters
    ----------
    X : array-like
        Array of features to transform.

    y : array-like
        Array of targets to transform.

    basename : str
        Store the precompzted transforms at `{basename}_X.npy` and `{basename}_y.npy`.

    transforms_X : list of callable, optional
        Ordered list of functions to call on `X`.

    transforms_y : list of callable, optional
        Ordered list of functions to call on `y`.

    logger : logging.Logger, optional
    """

    path_precomputed_data = basename + "_X.npy"
    path_precomputed_target = basename + "_y.npy"

    if (os.path.exists(path_precomputed_data) and os.path.exists(path_precomputed_target)):
        X = np.load(path_precomputed_data)
        y = np.load(path_precomputed_target)

    else:
        if logger is not None:
            logger.info("Precomputing preprocessed data ...")

        for transform_X in transforms_X:
            X = transform_X(X)

        for transform_y in transforms_y:
            y = transform_y(y)

        np.save(path_precomputed_data, X)
        np.save(path_precomputed_target, y)

    return X, y


# BATCH TRANSFORMS
def global_contrast_normalization(images, multiplier=55, eps=1e-10):
    """Performs global contrast normalization on a array or tensor of images.

    Notes
    -----
    - Compared to the notation in the the `Deep Learning` book(p 445), `lambda=0`
        `s = multiplier`.
    - modified from `https://github.com/brain-research/realistic-ssl-evaluation/`
    for reproducability.

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
    # Divide out the norm of each image (not std)
    per_image_norm = np.linalg.norm(images, axis=1, keepdims=True)
    # Avoid divide-by-zero
    per_image_norm[per_image_norm < eps] = 1
    images = multiplier * images / per_image_norm
    return images.reshape(*shape)


def get_zca_params(images, root_path, identity_scale=0.1, eps=1e-10, is_load=False):
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

    if is_load or (os.path.exists(mean_path) and os.path.exists(decomp_path)):
        image_mean = np.load(mean_path)
        zca_decomp = np.load(decomp_path)

    else:
        # never when testing
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

    return image_mean, zca_decomp


def zca_whitening(images, root, **kwargs):
    """
    Apply a ZCA normalization on a batch array and save the useds parameters.

    ZCA normalization is a type of whitening (linear transformation of features
    such that the covariance matrix becomes the identity => uncorrelated with
    variance=1 => becomes white noise). By removing 2 order correlation, it
    forces the mdoel to focus on high order correlation (multiple pixels) rather
    than "learning" that neighbouring pictures are similar.

    Notes
    -----
    - modified from `https://github.com/brain-research/realistic-ssl-evaluation/`
    for reproducability.

    Parameters
    ----------
    images : np.ndarray, shape = [batch_size, ...]
        Numpy array representing the original images.

    root_path : str
        Path to save the ZCA params to.

    kwargs :
        Additional arguments to `get_zca_params`.
    """
    image_mean, zca_decomp = get_zca_params(images, root, **kwargs)
    shape = images.shape
    images = images.reshape(shape[0], -1)
    return np.dot(images - image_mean, zca_decomp).reshape(*shape)


# SINGLE IMAGE TRANSFORMS
def random_translation(img, max_pix):
    """
    Random translations of 0 to max_pix given np.ndarray or PIL.Image(H W C).
    """
    is_pil = not isinstance(img, np.ndarray)
    if is_pil:
        img = np.atleast_3d(np.asarray(img))
    idx_h, idx_w = 0, 1
    img = np.pad(img, [[max_pix, max_pix], [max_pix, max_pix], [0, 0]],
                 mode="reflect")
    shifts = np.random.randint(-max_pix, max_pix + 1, size=[2])  # H and W
    processed_data = np.roll(img, shifts, (idx_h, idx_w))
    cropped_data = processed_data[max_pix:-max_pix, max_pix:-max_pix, :]
    if is_pil:
        img = Image.fromarray(img.squeeze())
    return cropped_data


def gaussian_noise(x, std=1, **kwargs):
    """Add Gaussian noise and clips the output."""
    return torch.clamp(x + torch.randn_like(x) * std, **kwargs)


class RobustMinMaxScaler(MinMaxScaler):
    """MinMaxScaling after clamping outliers."""

    def __init__(self, quantile=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.quantile = quantile

    def partial_fit(self, X, y=None):
        X = X.clip(np.quantile(X, self.quantile, axis=0, keepdims=True),
                   np.quantile(X, 1 - self.quantile, axis=0, keepdims=True))
        super().partial_fit(X, y=y)
        return self

    def transform(self, X):
        X = super().transform(X)
        # clipping before or after transform is same because lienar
        return X.clip(self.feature_range[0], self.feature_range[1])


def robust_minmax_scale(images, root_path, quantile=5e-4,
                        feature_range=(0, 1), is_load=False):

    scaler_path = os.path.join(root_path, "robust_scaler.npy")

    if is_load or os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = RobustMinMaxScaler(quantile=quantile, feature_range=feature_range)
        scaler.fit(images.reshape((-1, 1)))
        joblib.dump(scaler, scaler_path)

    return scaler.transform(images.reshape((-1, 1))).reshape(images.shape)
