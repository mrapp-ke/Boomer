#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for making predictions based on rules.
"""
from abc import abstractmethod

import numpy as np

from boomer.algorithm.model import Theory, DTYPE_FLOAT64
from boomer.interfaces import Randomized
from boomer.stats import Stats


class Prediction(Randomized):
    """
    A module that allows to make predictions using a 'Theory'.
    """

    @abstractmethod
    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        """
        Predicts the labels of examples using a specific theory.

        :param stats:   Statistics about the training data set
        :param theory:  The theory that is used to make predictions
        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        examples to be classified
        :return:        An array of dtype float, shape `(num_examples, num_labels)', representing the predicted labels
        """
        pass


class Ranking(Prediction):
    """
    A base class for all subclasses of the class 'Prediction' that predict numerical scores.
    """

    @abstractmethod
    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        pass


class LinearCombination(Ranking):
    """
    Predicts the linear combination of rules, i.e., the sum of the scores provided by all covering rules for each label.
    """

    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        predictions = np.asfortranarray(np.zeros((x.shape[0], stats.num_labels), dtype=DTYPE_FLOAT64))

        for rule in theory:
            rule.predict(x, predictions)

        return predictions


class Bipartition(Prediction):
    """
    A base class for all subclasses of the class 'Prediction' that predict binary label vectors.
    """

    @abstractmethod
    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        pass


class Sign(Bipartition):
    """
    Turns numerical scores into a binary label vector according to the sign function, i.e., 1, if a score is greater
    than zero, 1 otherwise.
    """

    def __init__(self, ranking: Ranking):
        """
        :param ranking: The ranking whose prediction should be turned into a binary label vector
        """
        self.ranking = ranking

    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        predictions = self.ranking.predict(stats, theory, x)
        return np.where(predictions > 0, 1, 0)
