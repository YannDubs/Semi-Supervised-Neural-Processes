import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def plot_2D_decision_boundary_SSL(X, y, model,
                                  title=None,
                                  mesh_stepsize=0.1,
                                  ax=None,
                                  scatter_kwargs={-1: {"c": "whitesmoke", "alpha": 0.4,
                                                       "linewidths": 0.5, "s": 10},
                                                  0: {"c": "b", "linewidths": 0.7, "s": 50},
                                                  1: {"c": "r", "linewidths": 0.7, "s": 50},
                                                  2: {"c": "g", "linewidths": 0.7, "s": 50}}):
    """Plot the 2D decision boundaries of a SSL classification model.

    Parameters
    ----------
    X: array-like
        2D input data

    y: array-like
        Labels, with `-1` for unlabeled points. Currently works with max 3 classes.

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
    def make_cmap(cmap_dflt, alpha=1):
        colors = cmap_dflt(np.linspace(0, 1, 256), alpha=alpha)
        cm = LinearSegmentedColormap.from_list('colormap', colors)
        cm.set_under(alpha=0)
        cm.set_over(alpha=0)
        return cm

    if ax is not None:
        F, ax = plt.subplots(1, 1, figsize=(7, 7))

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_stepsize),
                         np.arange(y_min, y_max, mesh_stepsize))

    if model is not None:
        try:
            y_hat = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])

            # for binary classification same as using plt.cm.RdBu
            cmaps = {0: make_cmap(plt.cm.Blues),
                     1: make_cmap(plt.cm.Reds),
                     2: make_cmap(plt.cm.Greens)}
            contourf_kwargs = dict(vmin=1 / (max(y) + 1), vmax=1, extend="neither", levels=10)

        except AttributeError:
            y_hat = model.predict(np.c_[xx.ravel(), yy.ravel()])
            contourf_kwargs = {"alpha": 1}
            is_3_class = max(y) == 2
            cmaps = {0: ListedColormap(["xkcd:darkish blue", "xkcd:darkish red"
                                        ] + (["xkcd:darkish green"] if is_3_class else []),
                                       name='my_colormap_name')}

        y_hat = y_hat.reshape(xx.shape + (-1,))

        for i in range(y_hat.shape[-1]):
            ax.contourf(xx, yy, y_hat[:, :, i], cmap=cmaps[i], **contourf_kwargs)

    for i in np.unique(y):
        idx = y == i
        ax.scatter(X[idx, 0], X[idx, 1],
                   marker="o", edgecolors="k", **scatter_kwargs[i])

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
