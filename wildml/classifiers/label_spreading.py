from sklearn.semi_supervised import label_propagation

__all__ = ["LabelSpreading"]


class LabelSpreading(label_propagation.LabelSpreading):
    __doc__ = (
        """
    Wrapper to change default hyperparameters and add `get_hypopt_search_space`.
    See documentation `sklearn.semi_supervised.LabelSpreading.
    Base documentation:
    """
        + label_propagation.LabelSpreading.__doc__
    )

    def __init__(
        self, kernel="knn", n_neighbors=1000, gamma=5, n_jobs=-1, max_iter=30, alpha=0.2, tol=0.001
    ):
        super().__init__(
            kernel=kernel,
            n_neighbors=n_neighbors,
            gamma=gamma,
            max_iter=max_iter,
            n_jobs=n_jobs,
            alpha=alpha,
            tol=tol,
        )

    def get_hypopt_search_space(self):
        """
        Return a good default search space as defined by `ConfigSpace`, this is easy to convert 
        to your favorite hyperparameter search library as well.
        """
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH

        config_space = CS.ConfigurationSpace()
        kernel = CSH.CategoricalHyperparameter(name="kernel", choices=["knn", "rbf"])
        n_neighbors = CSH.UniformIntegerHyperparameter(name="n_neighbors", lower=500, upper=3000)
        alpha = CSH.UniformFloatHyperparameter(
            "alpha", lower=1e-5, upper=1 - 1e-5, log=True, default=0.2
        )
        gamma = CSH.UniformFloatHyperparameter("gamma", lower=1e-3, upper=2e1, default=5, log=True)

        config_space.add_hyperparameter([kernel, gamma, alpha])

        cond_neighbors = CS.EqualsCondition(n_neighbors, kernel, "knn")
        cond_alpha = CS.EqualsCondition(alpha, kernel, "knn")
        cond_gamma = CS.EqualsCondition(gamma, kernel, "rbf")

        config_space.add_conditions([cond_neighbors, cond_alpha, cond_gamma])

        return config_space
