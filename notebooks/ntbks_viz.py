import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_dataset_samples(dataset, n_samples=50, title="Dataset", figsize=(8, 5)):
    """Plot `n_samples` samples of the a datset."""
    plt.figure(figsize=figsize)
    for i in range(n_samples):
        x, y = dataset[i]
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

    X_target = torch.Tensor(np.linspace(*min_max, n_trgt))
    X_target = X_target.unsqueeze(1).unsqueeze(0)

    plt.figure(figsize=figsize)
    for i in range(n_samples):
        z_sample = torch.randn((1, r_dim))
        p_y = model.decode(z_sample, X_target)

        X = X_target.numpy()[0].flatten()
        mean_y = p_y.base_dist.loc.detach().numpy()[0].flatten()
        std_y = p_y.base_dist.scale.detach().numpy()[0].flatten()

        plt.plot(X, mean_y, c='b', alpha=0.5)
        if is_plot_std:
            plt.fill_between(X, mean_y - std_y, mean_y + std_y)
        plt.xlim(*min_max)

    plt.title(title, fontsize=14)


def plot_posterior_samples(model, X_cntxt, Y_cntxt,
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

    X_target = torch.Tensor(np.linspace(*min_max, n_trgt))
    X_target = X_target.unsqueeze(1).unsqueeze(0)

    plt.figure(figsize=figsize)
    for i in range(n_samples):
        _, p_y_pred, _, _, _ = model.forward_step(X_cntxt, Y_cntxt, X_target)

        X = X_target.numpy()[0].flatten()
        mean_y = p_y_pred.base_dist.loc.detach().numpy()[0].flatten()
        std_y = p_y_pred.base_dist.scale.detach().numpy()[0].flatten()

        plt.plot(X, mean_y, alpha=0.05, c='b')

        if is_plot_std:
            plt.fill_between(X, mean_y - std_y, mean_y + std_y, alpha=0.05)
        plt.xlim(*min_max)

    print("std:", std_y.mean())
    plt.scatter(X_cntxt[0].numpy(), Y_cntxt[0].numpy(), c='k')

    plt.title(title, fontsize=14)
