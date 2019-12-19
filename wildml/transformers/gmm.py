import sklearn.mixture
from sklearn.base import TransformerMixin

__all__ = ["GaussianMixture"]


class GaussianMixture(sklearn.mixture.GaussianMixture, TransformerMixin):
    __doc__ = (
        """
    Wrapper to make GMM a transformer and add `get_hypopt_search_space`.
    See documentation `sklearn.mixture.GaussianMixture.
    Base documentation:
    """
        + sklearn.mixture.GaussianMixture.__doc__
    )

    def __init__(self, n_components=32, covariance_type="diag", **kwargs):
        super().__init__(n_components=32, covariance_type="diag", **kwargs)

    def transform(self, X):
        """Transform X to a the likelihood of being in any components.
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------
        X_new : array, shape = [n_samples, n_components]
            X transformed in the new space.
        """
        self._check_is_fitted()
        return self.predict_proba(X)

    def get_hypopt_search_space(self):
        """
        Return a good default search space as defined by `ConfigSpace`, this is easy to convert 
        to your favorite hyperparameter search library as well.
        """
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH

        config_space = CS.ConfigurationSpace()
        covariance_type = CSH.CategoricalHyperparameter(
            name="covariance_type", choices=["full", "diag"]
        )

        config_space.add_hyperparameter([covariance_type])

        return config_space
