import warnings

from sklearn.base import BaseEstimator, MetaEstimatorMixin
import numpy as np


class SelfTrainingMeta(MetaEstimatorMixin, BaseEstimator):
    """
    Turn any scikit learn estimator with a `predict_proba` method into a semi
    supervised one by iteratively training on labeled data and labellng confident
    predictions (above a threshold).

    Notes
    -----
    - Doesn't change labels of given data.
    - Reupdate all other labels at every iterations.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Base model to iteratively self train. Needs to have `predict_proba` class.

    max_iter : int, optional
        Maximum number of self-labeling iterations

    prob_threshold : float, optional
        Probability threshold to self-label

    weight_pred: float, optional
        Weight to give to newly labeled points.
    """

    def __init__(self, model,
                 max_iter=200,
                 p_threshold=0.8,
                 weight_pred=0.5):
        self.model = model
        self.max_iter = max_iter
        self.p_threshold = p_threshold
        self.weight_pred = weight_pred

        self.score = model.score
        self.predict = model.predict
        self.predict_proba = model.predict_proba

    def fit(self, X, y):
        """Fit base model to the data in a semi-supervised fashion
        using self training.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data.

        y : array_like, shape = [n_samples]
            Labels. Unlabeled points are marked as -1.
        """
        if (y == -1).sum() == 0:
            raise ValueError("Gave empty unlabeled data")

        X_unlabel = X[y == -1, :]
        X_label = X[y != -1, :]
        y_label = y[y != -1]

        self.model.fit(X_label, y_label)
        y_hat = self.predict(X_unlabel)
        p_y_hat = self.predict_proba(X_unlabel)
        is_no_change = False

        for i in range(self.max_iter):
            y_hat_old = y_hat
            is_conf = (p_y_hat > self.p_threshold).any(axis=1)

            samples_weight = [1 for _ in y_label] + [self.weight_pred for _ in y_hat[is_conf]]
            self.model.fit(np.concatenate((X_label, X_unlabel[is_conf, :]), axis=0),
                           np.concatenate((y_label, y_hat[is_conf]), axis=0),
                           sample_weight=samples_weight)
            y_hat = self.predict(X_unlabel)
            p_y_hat = self.predict_proba(X_unlabel)

            if (y_hat_old == y_hat).all():
                if is_no_change:
                    warnings.warn("Stopping self training after {} iterations, because no more changes.".format(i))
                    break
                else:
                    # only stop if twice in a row no change
                    is_no_change = True
            else:
                is_no_change = False

        return self
