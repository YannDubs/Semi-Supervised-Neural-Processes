import random

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import numpy as np

from wildml.utils.helpers import set_seed, prod

__all__ = ["plot_data_samples"]

DFLT_FIGSIZE = (17, 9)


def plot_data_samples(
    dataset, n_plots=4, figsize=DFLT_FIGSIZE, ax=None, pad_value=1, seed=123, title=None
):
    """Plot `n_samples` samples of the a datset."""
    set_seed(seed)

    dataset.rm_transformations()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    img_tensor = torch.stack(
        [dataset[random.randint(0, len(dataset) - 1)][0] for i in range(n_plots)], dim=0
    )
    grid = make_grid(img_tensor, nrow=2, pad_value=pad_value)

    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")

    if title is not None:
        ax.set_title(title, fontsize=18)

    dataset.set_transformations()
