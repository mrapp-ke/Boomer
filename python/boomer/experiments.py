#!/usr/bin/python


"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for performing experiments.
"""
import logging as log
from abc import ABC, abstractmethod

from sklearn.base import clone

from boomer.evaluation import Evaluation
from boomer.learners import MLLearner
from boomer.parameters import ParameterInput
from boomer.training import CrossValidation


class AbstractExperiment(CrossValidation, ABC):
    """
    An abstract base class for all experiments. It automatically encodes nominal attributes using one-hot encoding.
    """

    def __init__(self, evaluation: Evaluation, data_dir: str, data_set: str, num_folds: int, current_fold: int):
        """
        :param evaluation:      The evaluation to be used
        """

        super().__init__(data_dir, data_set, num_folds, current_fold)
        self.evaluation = evaluation

    @abstractmethod
    def _train_and_evaluate(self, train_indices, train_x, train_y, test_indices, test_x, test_y, first_fold: int,
                            current_fold: int, last_fold: int, num_folds: int):
        pass


class Experiment(AbstractExperiment):
    """
    An experiment that trains and evaluates a single multi-label classifier or ranker on a specific data set using cross
    validation or separate training and test sets.
    """

    def __init__(self, base_learner: MLLearner, evaluation: Evaluation, data_dir: str, data_set: str,
                 num_folds: int = 1, current_fold: int = -1, parameter_input: ParameterInput = None):
        """
        :param base_learner:    The classifier or ranker to be trained
        :param parameter_input: The input that should be used to read the parameter settings
        """
        super().__init__(evaluation, data_dir, data_set, num_folds, current_fold)
        self.base_learner = base_learner
        self.parameter_input = parameter_input

    def run(self):
        log.info('Starting experiment \"' + self.base_learner.get_name() + '\"...')
        super().run()

    def _train_and_evaluate(self, train_indices, train_x, train_y, test_indices, test_x, test_y, first_fold: int,
                            current_fold: int, last_fold: int, num_folds: int):
        base_learner = self.base_learner
        parameter_input = self.parameter_input

        # Apply parameter setting, if necessary
        current_learner = clone(base_learner)

        if parameter_input is not None:
            params = parameter_input.read_parameters(current_fold)
            current_learner.set_params(**params)
            log.info('Successfully applied parameter setting: %s', params)

        # Train classifier
        current_learner.random_state = self.random_state
        current_learner.fold = current_fold
        current_learner.fit(train_x, train_y)

        # Obtain and evaluate predictions for test data
        predictions = current_learner.predict(test_x)
        self.evaluation.evaluate(current_learner.get_name(), predictions, test_y, first_fold=first_fold,
                                 current_fold=current_fold, last_fold=last_fold, num_folds=num_folds)
