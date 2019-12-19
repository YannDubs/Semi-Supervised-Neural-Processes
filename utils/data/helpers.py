import torch

from wildml.utils.helpers import tmp_seed
from wildml.utils.datasplit import RandomMasker


def get_masks_drop_features(drop_size, mask_shape, n_masks, n_batch=32, seed=123):
    """
    Parameters
    ----------
    drop_size : float or int or tuple, optional
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to 
        drop. If int, represents the number of datapoints to drop. If tuple, same as before 
        but give bounds (min and max). 0 means keep all.

    mask_shape : tuple of int or callable
        Shape of the mask for one example. If callable, it is given the current index.

    n_masks : int, optional
        Number of masks to return.

    n_batch : int, optional
        Size of the batches of masks => number fo concsecutive examples with the same abount of 
        kept features.

    seed : int, optional
        Random seed.

    Returns
    -------
    to_drops : list of torch.BoolTensor
        List of length n_masks where each element is a boolean tensor of shape `mask_shape` with
        1s where features should be droped.

    Examples
    --------
    >>> get_masks_drop_features(0.5, (10,), 1, n_batch=1)
    [tensor([ True, False, False, False,  True,  True, False,  True,  True, False])]
    """

    try:
        mask_shape(0)
    except TypeError:

        def mask_shape(_, ret=mask_shape):
            return ret

    if drop_size == 0:
        return [torch.zeros(1, *mask_shape(i)).bool() for i in n_batch]

    with tmp_seed(seed):
        try:
            droper = RandomMasker(
                min_nnz=drop_size[0], max_nnz=drop_size[1], is_batch_repeat=False
            )
        except TypeError:
            droper = RandomMasker(min_nnz=drop_size, max_nnz=drop_size, is_batch_repeat=False)

        to_drops = []

        for i in range(0, n_masks, n_batch):
            to_drop = droper(n_batch, mask_shape(i))
            to_drops.extend(torch.unbind(to_drop.bool(), dim=0))

    return to_drops

