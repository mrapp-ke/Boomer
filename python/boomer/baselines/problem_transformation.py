#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Implements different problem transformation methods.
"""
from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.multioutput import ClassifierChain
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset

from boomer.learners import MLLearner, Learner
from boomer.stats import Stats


class ProblemTransformationLearner(MLLearner):
    """
    A base class for all multi-label classifiers that use a problem transformation method.
    """

    def __init__(self, model_dir: str, base_learner: Learner):
        """
        :param base_learner: The base learner to be used by the problem transformation method
        """
        super().__init__(model_dir)
        self.base_learner = base_learner

    def get_params(self, deep=True):
        params = super().get_params()
        params.update({
            'base_learner': self.base_learner
        })
        return params

    def set_params(self, **parameters):
        base_learner = self.base_learner
        base_learner_params = base_learner.get_params()
        keys_to_be_set = parameters.keys()
        base_learner_params_to_be_set = {}

        for key in base_learner_params.keys():
            if key in keys_to_be_set:
                value = parameters.pop(key)
                base_learner_params_to_be_set.update({key: value})

        if len(base_learner_params) > 0:
            base_learner.set_params(**base_learner_params_to_be_set)

        return super().set_params(**parameters)

    def _fit(self, stats: Stats, x: np.ndarray, y: np.ndarray, random_state: int):
        base_learner = self.base_learner
        transformation_method = self._create_transformation_method(base_learner, stats, x, y, random_state)
        transformation_method.fit(x, y)
        return transformation_method

    def _predict(self, model, stats: Stats, x: np.ndarray, random_state: int) -> np.ndarray:
        return model.predict(x)

    def get_name(self) -> str:
        return self.base_learner.get_name()

    @abstractmethod
    def _create_transformation_method(self, base_learner: Learner, stats: Stats, x: np.ndarray, y: np.ndarray,
                                      random_state: int) -> BaseEstimator:
        pass

    @abstractmethod
    def get_model_prefix(self) -> str:
        pass


class BRLearner(ProblemTransformationLearner):
    """
    A multi-label classifier that uses the binary relevance (BR) method.
    """

    def __init__(self, model_dir: str, base_learner: Learner):
        super().__init__(model_dir, base_learner)

    def _create_transformation_method(self, base_learner: Learner, stats: Stats, x: np.ndarray, y: np.ndarray,
                                      random_state: int) -> BaseEstimator:
        return BinaryRelevance(classifier=base_learner)

    def get_model_prefix(self) -> str:
        return 'br'


class LPLearner(ProblemTransformationLearner):
    """
    A multi-label classifier that uses the label powerset (LP) method.
    """

    def __init__(self, model_dir: str, base_learner: Learner):
        super().__init__(model_dir, base_learner)

    def _create_transformation_method(self, base_learner: Learner, stats: Stats, x: np.ndarray, y: np.ndarray,
                                      random_state: int) -> BaseEstimator:
        return LabelPowerset(classifier=base_learner)

    def get_model_prefix(self) -> str:
        return 'lp'


class CCLearner(ProblemTransformationLearner):
    """
    A multi-label classifier that uses a classifier chain (CC).
    """

    def __init__(self, model_dir: str, base_learner: Learner, chain_order: int = 1):
        super().__init__(model_dir, base_learner)
        self.chain_order = chain_order
        self.require_dense = [True, True]  # We need a dense representation of the training data

    def get_params(self, deep=True):
        params = super().get_params()
        params.update({
            'chain_order': self.chain_order
        })
        return params

    def _fit(self, stats: Stats, x: np.ndarray, y: np.ndarray, random_state: int):
        return super()._fit(stats, self._ensure_input_format(x), self._ensure_input_format(y), random_state)

    def _create_transformation_method(self, base_learner: Learner, stats: Stats, x: np.ndarray, y: np.ndarray,
                                      random_state: int) -> BaseEstimator:
        return ClassifierChain(base_learner, order='random', random_state=self.chain_order)

    def get_model_prefix(self) -> str:
        return 'cc'

    def get_name(self) -> str:
        return super().get_name() + '_chain-order=' + str(self.chain_order)
