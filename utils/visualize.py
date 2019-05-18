import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def plot_2D_decision_boundary_SSL(X, y, model,
                                  title=None,
                                  mesh_stepsize=.02):
    """Plot the 2D decision boundaries of a SSL classification model.

    Parameters
    ----------
    X : array-like
        2D input data

    y : array-like
        Labels, with `-1` for unlabeled points. Currently works with max 3 classes.

    model : sklearn.BaseEstimator
        Trained model.

    title : str, optional
        Title to add.

    mesh_stepsize : float, optional
        Step size of the mesh. Decrease to increase the quality.
    """
    def make_cmap(cmap_dflt, alpha=1):
        colors = cmap_dflt(np.linspace(0, 1, 256), alpha=alpha)
        cm = LinearSegmentedColormap.from_list('colormap', colors)
        cm.set_under(alpha=0)
        cm.set_over(alpha=0)
        return cm

    n_lab = (y != -1).sum()
    is_3_class = max(y) == 2
    cmap_nolab = ListedColormap(["whitesmoke"])
    cmap_lab = ListedColormap(["b", "r"] + (["g"] if is_3_class else []))

    F, ax = plt.subplots(1, 1, figsize=(7, 7))

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_stepsize),
                         np.arange(y_min, y_max, mesh_stepsize))

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
        cmaps = {0: ListedColormap(["xkcd:darkish blue", "xkcd:darkish red"
                                    ] + (["xkcd:darkish green"] if is_3_class else []),
                                   name='my_colormap_name')}

    y_hat = y_hat.reshape(xx.shape + (-1,))

    for i in range(y_hat.shape[-1]):
        ax.contourf(xx, yy, y_hat[:, :, i], cmap=cmaps[i], **contourf_kwargs)

    ax.scatter(X[n_lab:, 0], X[n_lab:, 1], s=10, marker="o", c=y[n_lab:],
               edgecolors="k", cmap=cmap_nolab, alpha=0.4, linewidths=0.5)

    ax.scatter(X[:n_lab, 0], X[:n_lab, 1], s=50, marker="o", c=y[:n_lab],
               edgecolors="k", cmap=cmap_lab, linewidths=0.7)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    return plt
