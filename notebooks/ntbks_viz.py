import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("..")

from skssl.utils.helpers import rescale_range


def plot_dataset_samples(dataset, n_samples=50, title="Dataset",
                         figsize=(8, 5)):
    """Plot `n_samples` samples of the a datset."""
    plt.figure(figsize=figsize)
    for i in range(n_samples):
        x, y = dataset[i]
        x = rescale_range(x, (-1, 1), dataset.min_max)
        plt.plot(x.numpy(), y.numpy(), c='b', alpha=0.5)
        plt.xlim(*dataset.min_max)
    plt.title(title, fontsize=14)


def plot_prior_samples(model, r_dim,
                       n_samples=50,
                       n_trgt=100,
                       is_plot_std=False,
                       title="Prior Samples",
                       figsize=(8, 5),
                       min_max=(-5, 5)):
    """
    Plot the mean at `n_trgt` different points for `n_samples`
    different latents (i.e. sampled functions).
    """
    model.eval()
    model = model.cpu()

    X_target = torch.Tensor(np.linspace(-1, 1, n_trgt))
    X_target = X_target.unsqueeze(1).unsqueeze(0)

    X_trgt_plot = X_target.numpy()[0].flatten()
    # input to model should always be between -1 1 but not for plotting
    X_trgt_plot = rescale_range(X_trgt_plot, (-1, 1), min_max)

    plt.figure(figsize=figsize)
    for i in range(n_samples):
        z_sample = torch.randn((1, r_dim))
        r = z_sample.unsqueeze(1).expand(1, n_trgt, r_dim)
        dec_input = model.make_dec_inp(r, z_sample, X_target)
        p_y = model.decode(dec_input)

        mean_y = p_y.base_dist.loc.detach().numpy()[0].flatten()
        std_y = p_y.base_dist.scale.detach().numpy()[0].flatten()

        plt.plot(X_trgt_plot, mean_y, c='b', alpha=0.5)
        if is_plot_std:
            plt.fill_between(X_trgt_plot, mean_y - std_y, mean_y + std_y)
        plt.xlim(*min_max)

    plt.title(title, fontsize=14)


def plot_posterior_samples(model, X_cntxt, Y_cntxt,
                           true_func=None,
                           n_samples=50,
                           n_trgt=100,
                           is_plot_std=False,
                           title="Posterior Samples Conditioned on Context",
                           figsize=(8, 5),
                           min_max=(-5, 5)):
    """
    Plot the mean at `n_trgt` different points for `n_samples`
    different latents (i.e. sampled functions) conditioned on cntxt points.
    """
    model.eval()
    model = model.cpu()

    X_target = torch.Tensor(np.linspace(-1, 1, n_trgt))
    X_target = X_target.unsqueeze(1).unsqueeze(0)

    # input to model should always be between -1 1 but not for plotting
    X_trgt_plot = X_target.numpy()[0].flatten()
    X_cntxt_plot = X_cntxt.numpy()[0].flatten()
    X_trgt_plot = rescale_range(X_trgt_plot, (-1, 1), min_max)
    X_cntxt_plot = rescale_range(X_cntxt_plot, (-1, 1), min_max)

    alpha = 1 / (n_samples + 1)**0.5

    plt.figure(figsize=figsize)
    for i in range(n_samples):
        p_y_pred, _, _, _ = model.forward_step(X_cntxt, Y_cntxt, X_target)

        mean_y = p_y_pred.base_dist.loc.detach().numpy()[0].flatten()
        std_y = p_y_pred.base_dist.scale.detach().numpy()[0].flatten()

        plt.plot(X_trgt_plot, mean_y, alpha=alpha, c='b')

        if is_plot_std:
            plt.fill_between(X_trgt_plot, mean_y - std_y, mean_y + std_y,
                             alpha=alpha / 7, color='tab:blue')
        plt.xlim(*min_max)

    if true_func is not None:
        X_true = true_func[0].numpy()[0].flatten()
        Y_true = true_func[1].numpy()[0].flatten()
        X_true = rescale_range(X_true, (-1, 1), min_max)
        plt.plot(X_true, Y_true, "--k", alpha=0.7)

    print("std:", std_y.mean())
    plt.scatter(X_cntxt_plot, Y_cntxt[0].numpy(), c='k')

    plt.title(title, fontsize=14)
