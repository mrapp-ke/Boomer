#!/usr/bin/python


"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for implementing single- or multi-label classifiers or rankers.
"""
import logging as log
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

    Attributes
        model_  The model
    """

    def fit(self, x, y):
        log.info('Fitting model...')
        start_time = timer()
        model = self._fit(x, y)
        end_time = timer()
        run_time = end_time - start_time
        log.info('Successfully fit model in %s seconds', run_time)
        self.model_ = model
        return self

    def predict(self, x):
        check_is_fitted(self)
        log.info("Making a prediction for %s query instances...", x.shape[0])
        return self._predict(x)

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

        :param x:   A numpy.ndarray or scipy.sparse matrix of shape `(num_examples, num_features)`, representing the
                    feature values of the training examples
        :param y:   A numpy.ndarray or scipy.sparse matrix of shape `(num_examples, num_labels)`, representing the
                    labels of the training examples
        :return:    The model that has been trained
        """
        pass

    @abstractmethod
    def _predict(self, x):
        """
        Makes a prediction for given query examples.

        :param x:   A numpy.ndarray or scipy.sparse matrix of shape `(num_examples, num_features)`, representing the
                    feature values of the query examples
        :return:    A numpy.ndarray or scipy.sparse matrix of shape `(num_examples, num_labels)`, representing the
                    labels predicted for the given query examples
        """
        pass
