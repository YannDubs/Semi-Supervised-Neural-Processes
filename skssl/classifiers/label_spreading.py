from sklearn.semi_supervised import label_propagation

from skopt.space import Real, Categorical, Integer


class LabelSpreading(label_propagation.LabelSpreading):
    __doc__ = """
    Wrapper to change default hyperparameters and add `get_hypopt_search_space`.
    See documentation `sklearn.semi_supervised.LabelSpreading.

    Base documentation:
    """ + label_propagation.LabelSpreading.__doc__

    def __init__(self, kernel="knn", n_neighbors=1000, gamma=5, n_jobs=-1,
                 max_iter=30, alpha=0.2, tol=0.001):
        super().__init__(kernel=kernel, n_neighbors=n_neighbors, gamma=gamma,
                         max_iter=max_iter, n_jobs=n_jobs, alpha=alpha, tol=tol)

    def get_hypopt_search_space(self):
        """Return a good default dearch space compatible with `skopt.BayesSearchCV`."""
        return [({'kernel': ['knn'],
                  'n_neighbors': Integer(500, 3000),
                  "alpha": Real(0, 1)}),
                ({'kernel': ['rbf'],
                  'gamma': Real(1e-3, 2e+1, prior='log-uniform'),
                  "alpha": Real(0, 1)})]
