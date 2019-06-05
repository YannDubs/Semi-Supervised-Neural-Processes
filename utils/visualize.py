from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def make_cmap(cmap_dflt, alpha=1):
    if isinstance(cmap_dflt, list):
        colors = cmap_dflt
    else:
        colors = cmap_dflt(np.linspace(0, 1, 256), alpha=alpha)
    cm = LinearSegmentedColormap.from_list('colormap', colors)
    cm.set_under(alpha=0)
    cm.set_over(alpha=0)
    return cm


def get_sequential_colors(n):
    """
    Return a list of n sequential color maps, the extreme color associated
    with it (or similar color) and a bright similar color.
    """
    assert n <= 10
    # for binary classification same as using plt.cm.RdBu
    cmaps = [make_cmap(plt.cm.Blues),
             make_cmap(plt.cm.Reds),
             make_cmap(plt.cm.Greens),
             make_cmap(plt.cm.Purples),
             make_cmap(plt.cm.Greys),
             make_cmap(plt.cm.Oranges),
             make_cmap(["white", "xkcd:olive"]),
             make_cmap(["white", "xkcd:brown"]),
             make_cmap(["white", "xkcd:dark turquoise"]),
             make_cmap(["white", "xkcd:bordeaux"])]

    extreme_colors = ["xkcd:darkish blue",
                      "xkcd:darkish red",
                      "xkcd:darkish green",
                      "xkcd:indigo",
                      "xkcd:black",
                      "xkcd:dark orange",
                      "xkcd:olive",
                      "xkcd:brown",
                      "xkcd:dark turquoise",
                      "xkcd:bordeaux"]

    bright_colors = ["xkcd:bright blue",
                     "xkcd:bright red",
                     "xkcd:green",
                     "xkcd:bright purple",
                     "k",
                     "xkcd:bright orange",
                     "xkcd:bright olive",
                     "xkcd:golden brown",
                     "xkcd:bright turquoise",
                     "xkcd:purple red"]

    return cmaps[:n], extreme_colors[:n], bright_colors[:n]


def plot_2D_decision_boundary_SSL(X, y, model,
                                  title=None,
                                  ax=None,
                                  mesh_stepsize=0.1,
                                  scatter_unlabelled_kwargs={"c": "whitesmoke",
                                                             "alpha": 0.4,
                                                             "linewidths": 0.5,
                                                             "s": 10},
                                  scatter_labelled_kwargs={"linewidths": 0.7, "s": 50}):
    """Plot the 2D decision boundaries of a SSL classification model.

    Parameters
    ----------
    X: array-like
        2D input data

    y: array-like
        Labels, with `-1` for unlabeled points. Currently works with max 10 classes.

    model: sklearn.BaseEstimator
        Trained model. If `None` plot the dataset only.

    title: str, optional
        Title to add.

    ax: matplotlib.axes, optional
        Axis on which to plot.

    mesh_stepsize: float, optional
        Step size of the mesh. Decrease to increase the quality. .02 is a good value
        for nice quality, 0.1 is faster but still ok.
    """
    if ax is None:
        F, ax = plt.subplots(1, 1, figsize=(7, 7))

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_stepsize),
                         np.arange(y_min, y_max, mesh_stepsize))

    cmaps, extreme_colors, bright_colors = get_sequential_colors(max(y) + 1)
    if model is not None:
        try:
            y_hat = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            contourf_kwargs = dict(vmin=1 / (max(y) + 1), vmax=1, extend="neither", levels=10)

        except AttributeError:
            y_hat = model.predict(np.c_[xx.ravel(), yy.ravel()])
            contourf_kwargs = {"alpha": 1}
            cmaps = [ListedColormap(extreme_colors)]

        y_hat = y_hat.reshape(xx.shape + (-1,))

        for i in range(y_hat.shape[-1]):
            ax.contourf(xx, yy, y_hat[:, :, i], cmap=cmaps[i], **contourf_kwargs)

    for i in np.unique(y):
        idx = y == i

        if i == -1:
            scatter_kwargs = scatter_unlabelled_kwargs
        else:
            scatter_kwargs = scatter_labelled_kwargs
            scatter_kwargs["c"] = bright_colors[i]

        ax.scatter(X[idx, 0], X[idx, 1],
                   marker="o", edgecolors="k", **scatter_kwargs)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
