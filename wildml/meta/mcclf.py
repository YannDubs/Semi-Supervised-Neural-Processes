import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin

from essl.training import NeuralNetClassifier


__all__ = ["MCClassifier"]


class MCClassifier(ClassifierMixin):
    """Class that merges a pretrained stochastic transformer with a classifier. It does this in an 
    ensembles of k models, to marginalize out the noise coming from the transformer.

    Parameters
    ----------
    stochastic_transformer : essl.training.NeuralNetTransformer
        Pretrained stochastic transformer that would give a different representation at every step.

    Classifier : sklearn.base.ClassifierMixIN
        Uninitialized classifier.

    k : int, optional
        Number of monte carlo samples / models to avergae.

    ensemble_kwargs : dict, optional
        Additional arguments to `VotingClassifier`. 

    classifier_kwargs : dict, optional
        Additional arguments to `NeuralNetClassifier`. 
    """

    def __init__(
        self, stochastic_transformer, Classifier, k=10, classifier_kwargs={}, voting="soft"
    ):
        stochastic_transformer.freeze(True)
        self.estimators = [
            (
                f"{i}",
                Pipeline(
                    [
                        ("transformer", stochastic_transformer),
                        ("classifier", Classifier(**classifier_kwargs)),
                    ]
                ),
            )
            for i in range(k)
        ]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        avg : array-like, shape (n_samples, n_classes)
            Weighted average probability for each class per sample.
        """
        probs = np.asarray([clf.predict_proba(X) for n, clf in self.estimators])
        return np.average(probs, axis=0)

    def fit(self, X, y):
        names, clfs = zip(*self.estimators)
        for clf in clfs:
            clf.fit(X, y)
        return self
