#!/usr/bin/python


"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for performing experiments.
"""
import logging as log
from abc import ABC
from timeit import default_timer as timer

from mlrl.common.learners import Learner, NominalAttributeLearner
from mlrl.testbed.data import MetaData, AttributeType
from mlrl.testbed.data_characteristics import DataCharacteristicsPrinter
from mlrl.testbed.evaluation import Evaluation
from mlrl.testbed.model_characteristics import ModelPrinter, ModelCharacteristicsPrinter
from mlrl.testbed.parameters import ParameterInput
from mlrl.testbed.persistence import ModelPersistence
from mlrl.testbed.training import CrossValidation, DataSet
from sklearn.base import clone


class Experiment(CrossValidation, ABC):
    """
    An experiment that trains and evaluates a single multi-label classifier or ranker on a specific data set using cross
    validation or separate training and test sets.
    """

    def __init__(self, base_learner: Learner, data_set: DataSet, num_folds: int = 1, current_fold: int = -1,
                 train_evaluation: Evaluation = None, test_evaluation: Evaluation = None,
                 parameter_input: ParameterInput = None, model_printer: ModelPrinter = None,
                 model_characteristics_printer: ModelCharacteristicsPrinter = None,
                 data_characteristics_printer: DataCharacteristicsPrinter = None, persistence: ModelPersistence = None):
        """
        :param base_learner:                    The classifier or ranker to be trained
        :param train_evaluation:                The evaluation to be used for evaluating the predictions for the
                                                training data or None, if the predictions should not be evaluated
        :param test_evaluation:                 The evaluation to be used for evaluating the predictions for the test
                                                data or None, if the predictions should not be evaluated
        :param parameter_input:                 The input that should be used to read the parameter settings
        :param model_printer:                   The printer that should be used to print textual representations of
                                                models or None, if no textual representations should be printed
        :param model_characteristics_printer:   The printer that should be used to print the characteristics of models
                                                or None, if the characteristics should not be printed
        :param data_characteristics_printer:    The printer that should be used to print the characteristics of the
                                                training data or None, if the characteristics should not be printed
        :param persistence:                     The `ModelPersistence` that should be used for loading and saving models
        """
        super().__init__(data_set, num_folds, current_fold)
        self.base_learner = base_learner
        self.train_evaluation = train_evaluation
        self.test_evaluation = test_evaluation
        self.parameter_input = parameter_input
        self.model_printer = model_printer
        self.model_characteristics_printer = model_characteristics_printer
        self.data_characteristics_printer = data_characteristics_printer
        self.persistence = persistence

    def run(self):
        log.info('Starting experiment \"' + self.base_learner.get_name() + '\"...')
        super().run()

    def _train_and_evaluate(self, meta_data: MetaData, train_indices, train_x, train_y, test_indices, test_x, test_y,
                            first_fold: int, current_fold: int, last_fold: int, num_folds: int):
        base_learner = self.base_learner
        current_learner = clone(base_learner)

        # Apply parameter setting, if necessary...
        parameter_input = self.parameter_input

        if parameter_input is not None:
            params = parameter_input.read_parameters(current_fold)
            current_learner.set_params(**params)
            log.info('Successfully applied parameter setting: %s', params)

        learner_name = current_learner.get_name()

        # Print data characteristics, if necessary...
        data_characteristics_printer = self.data_characteristics_printer

        if data_characteristics_printer is not None:
            data_characteristics_printer.print(learner_name, train_x, train_y, meta_data, current_fold=current_fold,
                                               num_folds=num_folds)

        # Set the indices of nominal attributes, if supported...
        if isinstance(current_learner, NominalAttributeLearner):
            current_learner.nominal_attribute_indices = meta_data.get_attribute_indices(AttributeType.NOMINAL)

        # Load model from disc, if possible, otherwise train a new model...
        loaded_learner = self.__load_model(model_name=learner_name, current_fold=current_fold, num_folds=num_folds)

        if isinstance(loaded_learner, Learner):
            current_learner = loaded_learner
        else:
            log.info('Fitting model to %s training examples...', train_x.shape[0])
            current_learner.fit(train_x, train_y)
            log.info('Successfully fit model in %s seconds', current_learner.train_time_)

            # Save model to disk...
            self.__save_model(current_learner, current_fold=current_fold, num_folds=num_folds)

        # Obtain and evaluate predictions for training data, if necessary...
        evaluation = self.train_evaluation

        if evaluation is not None:
            log.info('Predicting for %s training examples...', train_x.shape[0])
            start_time = timer()
            predictions = current_learner.predict(train_x)
            end_time = timer()
            predict_time = end_time - start_time
            log.info('Successfully predicted in %s seconds', predict_time)
            evaluation.evaluate('train_' + learner_name, meta_data, predictions, train_y, first_fold=first_fold,
                                current_fold=current_fold, last_fold=last_fold, num_folds=num_folds,
                                train_time=current_learner.train_time_, predict_time=predict_time)

        # Obtain and evaluate predictions for test data, if necessary...
        evaluation = self.test_evaluation

        if evaluation is not None:
            log.info('Predicting for %s test examples...', test_x.shape[0])
            start_time = timer()
            predictions = current_learner.predict(test_x)
            end_time = timer()
            predict_time = end_time - start_time
            log.info('Successfully predicted in %s seconds', predict_time)
            evaluation.evaluate('test_' + learner_name, meta_data, predictions, test_y, first_fold=first_fold,
                                current_fold=current_fold, last_fold=last_fold, num_folds=num_folds,
                                train_time=current_learner.train_time_, predict_time=predict_time)

        # Print model characteristics, if necessary...
        model_characteristics_printer = self.model_characteristics_printer

        if model_characteristics_printer is not None:
            model_characteristics_printer.print(learner_name, current_learner, current_fold=current_fold,
                                                num_folds=num_folds)

        # Print model, if necessary...
        model_printer = self.model_printer

        if model_printer is not None:
            model_printer.print(learner_name, meta_data, current_learner, current_fold=current_fold,
                                num_folds=num_folds)

    def __load_model(self, model_name: str, current_fold: int, num_folds: int):
        """
        Loads the model from disk, if available.

        :param model_name:      The name of the model to be loaded
        :param current_fold:    The current fold starting at 0, or 0 if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        :return: The loaded model
        """
        persistence = self.persistence

        if persistence is not None:
            return persistence.load_model(model_name=model_name, fold=(current_fold if num_folds > 1 else None))

        return None

    def __save_model(self, model: Learner, current_fold: int, num_folds: int):
        """
        Saves a model to disk.

        :param model:           The model to be saved
        :param current_fold:    The current fold starting at 0, or 0 if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        """
        persistence = self.persistence

        if persistence is not None:
            persistence.save_model(model, model_name=model.get_name(), fold=(current_fold if num_folds > 1 else None))
