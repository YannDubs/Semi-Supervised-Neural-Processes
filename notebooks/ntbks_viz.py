import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("..")

from skssl.utils.helpers import rescale_range

DFLT_FIGSIZE = (11, 5)


def plot_dataset_samples(dataset, n_samples=50, title="Dataset",
                         figsize=DFLT_FIGSIZE):
    """Plot `n_samples` samples of the a datset."""
    plt.figure(figsize=figsize)
    alpha = 1 / (n_samples + 1)**0.5

    for i in range(n_samples):
        x, y = dataset[i]
        x = rescale_range(x, (-1, 1), dataset.min_max)
        plt.plot(x.numpy(), y.numpy(), c='b', alpha=alpha)
        plt.xlim(*dataset.min_max)
    plt.title(title, fontsize=14)


def _rescale_ylim(y_min, y_max):
    """Make the y_lim range larger."""
    if y_min < 0:
        y_min *= 1.2
    else:
        y_min /= 0.9

    if y_max > 0:
        y_max *= 1.2
    else:
        y_max /= 0.9
    return y_min, y_max


def _get_p_y_pred(model, X_cntxt, Y_cntxt, X_target):
    if X_cntxt is not None:
        p_y_pred, _, _, _ = model.forward_step(X_cntxt, Y_cntxt, X_target)
    else:
        z_sample = torch.randn((1, model.r_dim))
        r = z_sample.unsqueeze(1).expand(1, X_target.size(1), model.r_dim)
        dec_input = model.make_dec_inp(r, z_sample, X_target)
        p_y_pred = model.decode(dec_input, X_target)
    return p_y_pred


def plot_posterior_predefined_cntxt(model,
                                    X_cntxt=None,
                                    Y_cntxt=None,
                                    true_func=None,
                                    n_trgt=100,
                                    n_samples=50,
                                    is_plot_std=False,
                                    title=None,
                                    figsize=DFLT_FIGSIZE,
                                    train_min_max=(-5, 5),
                                    test_min_max=None):
    """
    Plot the mean at `n_trgt` different points for `n_samples` different
     latents (i.e. sampled functions) conditioned on some predefined cntxt points.
    """
    is_extrapolating = test_min_max is not None
    if not is_extrapolating:
        test_min_max = train_min_max

    is_conditioned = X_cntxt is not None  # plot posterior instead prior

    model.eval()
    model = model.cpu()

    # scale such that interpolation is in [-1,1] but extrapolation will not
    input_min_max = tuple(rescale_range(np.array(test_min_max), train_min_max, (-1, 1)))
    X_target = torch.Tensor(np.linspace(*input_min_max, n_trgt))
    X_target = X_target.view(1, -1, 1)

    X_trgt_plot = X_target.numpy()[0].flatten()
    X_interp = (X_trgt_plot > -1) & (X_trgt_plot < 1)
    # input to model should always be between -1 1 but not for plotting
    X_trgt_plot = rescale_range(X_trgt_plot, (-1, 1), train_min_max)

    if is_conditioned:
        X_cntxt_plot = X_cntxt.numpy()[0].flatten()
        X_cntxt_plot = rescale_range(X_cntxt_plot, (-1, 1), train_min_max)

    alpha = 1 / (n_samples + 1)**0.5

    y_min = 0
    y_max = 0
    std_y_mean = 0
    plt.figure(figsize=figsize)
    for i in range(n_samples):
        p_y_pred = _get_p_y_pred(model, X_cntxt, Y_cntxt, X_target)

        mean_y = p_y_pred.base_dist.loc.detach().numpy()[0].flatten()
        std_y = p_y_pred.base_dist.scale.detach().numpy()[0].flatten()
        std_y_mean += std_y.mean() / n_samples

        plt.plot(X_trgt_plot, mean_y, alpha=alpha, c='b')

        if is_plot_std:
            plt.fill_between(X_trgt_plot, mean_y - std_y, mean_y + std_y,
                             alpha=alpha / 7, color='tab:blue')
            y_min = min(y_min, (mean_y - std_y)[X_interp].min())
            y_max = max(y_max, (mean_y + std_y)[X_interp].max())
        else:
            y_min = min(y_min, (mean_y)[X_interp].min())
            y_max = max(y_max, (mean_y)[X_interp].max())

        plt.xlim(*test_min_max)

    if true_func is not None:
        X_true = true_func[0].numpy()[0].flatten()
        Y_true = true_func[1].numpy()[0].flatten()
        X_true = rescale_range(X_true, (-1, 1), train_min_max)
        plt.plot(X_true, Y_true, "--k", alpha=0.7)
        y_min = min(y_min, Y_true.min())
        y_max = max(y_max, Y_true.max())

    print("std:", std_y_mean)

    if is_conditioned:
        plt.scatter(X_cntxt_plot, Y_cntxt[0].numpy(), c='k')

    # extrapolation might give huge values => rescale to have y lim as interpolation
    plt.ylim(_rescale_ylim(y_min, y_max))

    if is_extrapolating:
        plt.axvline(x=train_min_max[1], color='r', linestyle=':', alpha=0.5,
                    label='Interpolation-Extrapolation boundary')
        plt.legend()

    if title is not None:
        plt.title(title, fontsize=14)


def plot_prior_samples(model, title="Prior Samples", **kwargs):
    """
    Plot the mean at `n_trgt` different points for `n_samples`
    different latents (i.e. sampled functions).
    """
    plot_posterior_predefined_cntxt(model, X_cntxt=None, Y_cntxt=None,
                                    true_func=None, title=title, **kwargs)


def plot_posterior_samples(dataset, model,
                           n_cntxt=10,
                           n_points=None,
                           is_plot_std=True,
                           test_min_max=None,
                           is_force_cntxt_extrap=False,
                           **kwargs):
    """
    Plot the mean at `n_trgt` different points samples if `test_min_max` for
    `n_samples` different latents  (i.e. sampled functions) conditioned on `n_cntxt`
    context point sampeld from  a dataset.
    """
    if n_points is None:
        n_points = dataset.n_points
    X, Y = dataset.extrapolation_samples(n_samples=1,
                                         test_min_max=test_min_max,
                                         n_points=n_points)
    # randomly subset for context
    idcs = torch.randperm(n_points)[:n_cntxt]

    if is_force_cntxt_extrap and test_min_max is not None:
        # dirty trick to always get one value at extrapolation (hope that 5 before
        # last is extrapolation)
        idcs = torch.cat((idcs, torch.tensor([n_points - 5])))

    X_cntxt, Y_cntxt = X[:, idcs, :], Y[:, idcs, :]
    plot_posterior_predefined_cntxt(model, X_cntxt, Y_cntxt,
                                    true_func=(X, Y),
                                    is_plot_std=is_plot_std,
                                    train_min_max=dataset.min_max,
                                    test_min_max=test_min_max,
                                    n_trgt=n_points,
                                    **kwargs)
