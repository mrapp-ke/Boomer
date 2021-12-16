#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for implementing single- or multi-label classifiers or rankers.
"""
from abc import ABC, abstractmethod
from timeit import default_timer as timer
from typing import List

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class NominalAttributeLearner(ABC):
    """
    A base class for all single- or multi-label classifiers or rankers that natively support nominal attributes.
    """

    nominal_attribute_indices: List[int] = None


class Learner(BaseEstimator):
    """
    A base class for all single- or multi-label classifiers or rankers.
    """

    def fit(self, x, y):
        """
        Fits a model according to given training examples and corresponding ground truth labels.

        :param x:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores the
                    feature values of the training examples
        :param y:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the
                    labels of the training examples according to the ground truth
        :return:    The fitted learner
        """
        start_time = timer()
        model = self._fit(x, y)
        end_time = timer()
        run_time = end_time - start_time
        self.model_ = model
        self.train_time_ = run_time
        return self

    def predict(self, x):
        """
        Makes a prediction for given query examples.

        :param x:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores the
                    feature values of the query examples
        :return:    A `numpy.ndarray` or `scipy.sparse` matrix of shape `(num_examples, num_labels)`, that stores the
                    prediction for individual examples and labels
        """
        check_is_fitted(self)
        return self._predict(x)

    def predict_proba(self, x):
        """
        Returns probability estimates for given query examples.

        :param x:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores the
                    feature values of the query examples
        :return:    A `numpy.ndarray` or `scipy.sparse` matrix of shape `(num_examples, num_labels)`, that stores the
                    probabilities for individual examples and labels
        """
        check_is_fitted(self)
        return self._predict_proba(x)

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns a human-readable name that allows to identify the configuration used by the classifier or ranker.

        :return: The name of the classifier or ranker
        """
        pass

    @abstractmethod
    def _fit(self, x, y):
        """
        Trains a new model on the given training data.

        :param x:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores the
                    feature values of the training examples
        :param y:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the
                    labels of the training examples according to the ground truth
        :return:    The model that has been trained
        """
        pass

    @abstractmethod
    def _predict(self, x):
        """
        Makes a prediction for given query examples.

        :param x:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores the
                    feature values of the query examples
        :return:    A `numpy.ndarray` or `scipy.sparse` matrix of shape `(num_examples, num_labels)`, that stores the
                    prediction for individual examples and labels
        """
        pass

    def _predict_proba(self, x):
        """
        Returns probability estimates for given query examples.

        :param x:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores the
                    feature values of the query examples
        :return:    A `numpy.ndarray` or `scipy.sparse` matrix of shape `(num_examples, num_labels)`, that stores the
                    probabilities for individual examples and labels
        """
        raise RuntimeError('Prediction of probabilities not supported using the current configuration')
