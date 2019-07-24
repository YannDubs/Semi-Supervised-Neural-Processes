import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import sklearn

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


def _get_p_y_pred(model, X_cntxt, Y_cntxt, X_target, **kwargs):
    if X_cntxt is not None:
        p_y_pred, *_ = model.forward_step(X_cntxt, Y_cntxt, X_target, **kwargs)
    else:
        z_sample = torch.randn((1, model.r_dim))
        r = z_sample.unsqueeze(1).expand(1, X_target.size(1), model.r_dim)
        dec_input = model.make_dec_inp(r, z_sample, X_target)
        p_y_pred = model.decode(dec_input, X_target, **kwargs)
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
                                    test_min_max=None,
                                    model_label="Model",
                                    ax=None,
                                    is_comparing=False,
                                    alpha_init=1,
                                    y_idx=0,
                                    is_sparse_channels=False,
                                    get_cntxt_trgt=None):
    """
    Plot the mean at `n_trgt` different points for `n_samples` different
     latents (i.e. sampled functions) conditioned on some predefined cntxt points.
    """
    if is_sparse_channels:
        if X_cntxt is not None:
            channels = X_cntxt[..., 0]
            mask = channels == y_idx
            X_cntxt = X_cntxt[mask][..., 1:].unsqueeze(0)  # dropt sparse channels
            Y_cntxt = Y_cntxt[mask].unsqueeze(0)

        if true_func is not None:
            true_func = list(true_func)  # support mutation
            channels = true_func[0][..., 0]
            mask = channels == y_idx
            true_func[0] = true_func[0][mask][..., 1:].unsqueeze(0)  # dropt sparse channels
            true_func[1] = true_func[1][mask].unsqueeze(0)

    if get_cntxt_trgt is not None:
        X_cntxt, Y_cntxt, _, _ = get_cntxt_trgt(true_func[0], true_func[1])

    if is_comparing:
        mean_color = "m"
        std_color = 'tab:pink'
    else:
        mean_color = "b"
        std_color = 'tab:blue'

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

    alpha = alpha_init / (n_samples)**0.5

    y_min = 0
    y_max = 0

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i in range(n_samples):
        if is_sparse_channels:
            # adds sparse channel for predictions
            X_cntxt_pred = torch.cat([torch.ones_like(X_cntxt).float() * y_idx, X_cntxt], dim=-1)
            X_target_pred = torch.cat([torch.ones_like(X_target).float() * y_idx, X_target], dim=-1)
        else:
            X_cntxt_pred, X_target_pred = X_cntxt, X_target

        p_y_pred = _get_p_y_pred(model, X_cntxt_pred, Y_cntxt, X_target_pred)  # , true_func[1])

        mean_y = p_y_pred.base_dist.loc.detach().numpy()[0, :, y_idx].flatten()
        std_y = p_y_pred.base_dist.scale.detach().numpy()[0, :, y_idx].flatten()

        if i == 0:
            ax.plot(X_trgt_plot, mean_y, alpha=alpha, c=mean_color,
                    label="{} Predictions".format(model_label))
        else:
            ax.plot(X_trgt_plot, mean_y, alpha=alpha, c=mean_color)

        if is_plot_std:
            ax.fill_between(X_trgt_plot, mean_y - std_y, mean_y + std_y,
                            alpha=alpha / 7, color=std_color)
            y_min = min(y_min, (mean_y - std_y)[X_interp].min())
            y_max = max(y_max, (mean_y + std_y)[X_interp].max())
        else:
            y_min = min(y_min, (mean_y)[X_interp].min())
            y_max = max(y_max, (mean_y)[X_interp].max())

        ax.set_xlim(*test_min_max)

    if true_func is not None and not is_comparing:
        X_true = true_func[0].numpy()[0].flatten()
        if is_sparse_channels:
            Y_true = true_func[1].numpy().flatten()
        else:
            Y_true = true_func[1].numpy()[0, :, y_idx].flatten()
        X_true = rescale_range(X_true, (-1, 1), train_min_max)
        idx = np.argsort(X_true)
        ax.plot(X_true[idx], Y_true[idx], "--k", alpha=0.7, label='Sampled target function')
        y_min = min(y_min, Y_true.min())
        y_max = max(y_max, Y_true.max())

    if is_conditioned:
        if is_sparse_channels:
            Y_cntxt = Y_cntxt.numpy().flatten()
        else:
            Y_cntxt = Y_cntxt[0, :, y_idx].numpy()
        ax.scatter(X_cntxt_plot, Y_cntxt, c='k')

    # extrapolation might give huge values => rescale to have y lim as interpolation
    ax.set_ylim(_rescale_ylim(y_min, y_max))

    if is_extrapolating and not is_comparing:
        ax.axvline(x=train_min_max[1], color='r', linestyle=':', alpha=0.5,
                   label='Interpolation-Extrapolation boundary')

    if title is not None and not is_comparing:
        ax.set_title(title, fontsize=14)

    ax.legend()

    return ax


def plot_prior_samples(model, title="Prior Samples", **kwargs):
    """
    Plot the mean at `n_trgt` different points for `n_samples`
    different latents (i.e. sampled functions).
    """
    ax = plot_posterior_predefined_cntxt(model, X_cntxt=None, Y_cntxt=None,
                                         true_func=None, title=title, **kwargs)


def plot_posterior_samples(dataset, model,
                           compare_model=None,
                           model_labels=["Model", "Compare"],
                           n_cntxt=10,
                           n_points=None,
                           is_plot_std=True,
                           test_min_max=None,
                           is_force_cntxt_extrap=False,
                           is_plot_generator=True,
                           is_true_func=True,
                           **kwargs):
    """
    Plot the mean at `n_trgt` different points samples if `test_min_max` for
    `n_samples` different latents  (i.e. sampled functions) conditioned on `n_cntxt`
    context point sampeld from  a dataset.
    """
    if n_points is None:
        n_points = dataset.n_points
    X, Y = dataset.get_samples(n_samples=1,
                               test_min_max=test_min_max,
                               n_points=n_points)
    # randomly subset for context
    idcs = torch.randperm(n_points)[:n_cntxt]

    if is_force_cntxt_extrap and test_min_max is not None:
        # dirty trick to always get one value at extrapolation (hope that 5 before
        # last is extrapolation)
        idcs = torch.cat((idcs, torch.tensor([n_points - 5])))

    X_cntxt, Y_cntxt = X[:, idcs, :], Y[:, idcs, :]

    alpha_init = 1 if compare_model is None else 0.5

    ax = plot_posterior_predefined_cntxt(model, X_cntxt, Y_cntxt,
                                         true_func=(X, Y) if is_true_func else None,
                                         is_plot_std=is_plot_std,
                                         train_min_max=dataset.min_max,
                                         test_min_max=test_min_max,
                                         n_trgt=n_points,
                                         model_label=model_labels[0],
                                         alpha_init=alpha_init,
                                         **kwargs)

    if compare_model is not None:
        ax = plot_posterior_predefined_cntxt(compare_model, X_cntxt, Y_cntxt,
                                             is_plot_std=is_plot_std,
                                             train_min_max=dataset.min_max,
                                             test_min_max=test_min_max,
                                             n_trgt=n_points,
                                             model_label=model_labels[1],
                                             ax=ax,
                                             is_comparing=True,
                                             alpha_init=alpha_init,
                                             **kwargs)

    if is_plot_generator:
        X_cntxt_plot = rescale_range(X_cntxt, (-1, 1), dataset.min_max).numpy()[0]
        # clones so doesn't change real generator => can still sample prior
        generator = sklearn.base.clone(dataset.generator)
        generator.fit(X_cntxt_plot, Y_cntxt.numpy()[0])
        X_trgt_plot = rescale_range(X, (-1, 1), dataset.min_max).numpy()[0].flatten()
        mean_y, std_y = generator.predict(X_trgt_plot[:, np.newaxis], return_std=True)
        mean_y = mean_y.flatten()
        ax.plot(X_trgt_plot, mean_y, alpha=alpha_init / 2, c="g", label="Generator's predictions")
        ax.fill_between(X_trgt_plot, mean_y - std_y, mean_y + std_y,
                        alpha=alpha_init / 10, color='tab:green')
        ax.legend()
