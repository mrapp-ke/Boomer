"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for performing experiments.
"""
import logging as log

from abc import ABC, abstractmethod
from functools import reduce
from timeit import default_timer as timer
from typing import List, Optional

from sklearn.base import BaseEstimator, RegressorMixin, clone

from mlrl.common.learners import IncrementalLearner, Learner, NominalAttributeLearner, OrdinalAttributeLearner

from mlrl.testbed.data import AttributeType, MetaData
from mlrl.testbed.data_splitting import DataSplit, DataSplitter, DataType
from mlrl.testbed.format import format_duration
from mlrl.testbed.output_writer import OutputWriter
from mlrl.testbed.parameters import ParameterInput
from mlrl.testbed.persistence import ModelPersistence
from mlrl.testbed.prediction_scope import GlobalPrediction, IncrementalPrediction, PredictionScope, PredictionType


class Evaluation(ABC):
    """
    An abstract base class for all classes that allow to evaluate predictions that are obtained from a previously
    trained model.
    """

    def __init__(self, prediction_type: PredictionType, output_writers: List[OutputWriter]):
        """
        :param prediction_type: The type of the predictions to be obtained
        :param output_writers:  A list that contains all output writers to be invoked after predictions have been
                                obtained
        """
        self.prediction_type = prediction_type
        self.output_writers = output_writers

    def _invoke_prediction_function(self, learner, predict_function, predict_proba_function, x):
        """
        May be used by subclasses in order to invoke the correct prediction function, depending on the type of
        result that should be obtained.

        :param learner:                 The learner, the result should be obtained from
        :param predict_function:        The function to be invoked if binary result or regression scores should be
                                        obtained
        :param predict_proba_function:  The function to be invoked if probability estimates should be obtained
        :param x:                       A `numpy.ndarray` or `scipy.sparse` matrix, shape
                                        `(num_examples, num_features)`, that stores the feature values of the query
                                        examples
        :return:                        The return value of the invoked function
        """
        prediction_type = self.prediction_type

        if prediction_type == PredictionType.SCORES:
            try:
                if isinstance(learner, Learner):
                    result = predict_function(x, predict_scores=True)
                elif isinstance(learner, RegressorMixin):
                    result = predict_function(x)
                else:
                    raise RuntimeError()
            except RuntimeError:
                log.error('Prediction of regression scores not supported')
                result = None
        elif prediction_type == PredictionType.PROBABILITIES:
            try:
                result = predict_proba_function(x)
            except RuntimeError:
                log.error('Prediction of probabilities not supported')
                result = None
        else:
            result = predict_function(x)

        return result

    def _evaluate_predictions(self, meta_data: MetaData, data_split: DataSplit, data_type: DataType,
                              prediction_scope: PredictionScope, train_time: float, predict_time: float, x, y,
                              predictions, learner):
        """
        May be used by subclasses in order to evaluate predictions that have been obtained from a previously trained
        model.

        :param meta_data:           The meta-data of the data set
        :param data_split:          The split of the available data, the predictions and ground truth labels correspond
                                    to
        :param data_type:           Specifies whether the predictions and ground truth labels correspond to the training
                                    or test data
        :param prediction_scope:    Specifies whether the predictions have been obtained from a global model or
                                    incrementally
        :param train_time:          The time needed to train the model
        :param predict_time:        The time needed to obtain the predictions
        :param x:                   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`,
                                    that stores the feature values of the query examples
        :param y:                   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                    stores the ground truth labels of the query examples
        :param predictions:         A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                    stores the predictions for the query examples
        :param learner:             The learner, the predictions have been obtained from
        """
        for output_writer in self.output_writers:
            output_writer.write_output(meta_data, x, y, data_split, learner, data_type, self.prediction_type,
                                       prediction_scope, predictions, train_time, predict_time)

    @abstractmethod
    def predict_and_evaluate(self, meta_data: MetaData, data_split: DataSplit, data_type: DataType, train_time: float,
                             learner, x, y):
        """
        Must be implemented by subclasses in order to obtain and evaluate predictions for given query examples from a
        previously trained model.

        :param meta_data:   The meta-data of the data set
        :param data_split:  The split of the available data, the predictions and ground truth labels correspond to
        :param data_type:   Specifies whether the predictions and ground truth labels correspond to the training or test
                            data
        :param train_time:  The time needed to train the model
        :param learner:     The learner, the predictions should be obtained from
        :param x:           A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that
                            stores the feature values of the query examples
        :param y:           A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores
                            the ground truth labels of the query examples
        """
        pass


class GlobalEvaluation(Evaluation):
    """
    Obtains and evaluates predictions from a previously trained global model.
    """

    def __init__(self, prediction_type: PredictionType, output_writers: List[OutputWriter]):
        super().__init__(prediction_type, output_writers)

    def predict_and_evaluate(self, meta_data: MetaData, data_split: DataSplit, data_type: DataType, train_time: float,
                             learner, x, y):
        log.info('Predicting for %s ' + data_type.value + ' examples...', x.shape[0])
        start_time = timer()
        predictions = self._invoke_prediction_function(learner, learner.predict, learner.predict_proba, x)
        end_time = timer()
        predict_time = end_time - start_time

        if predictions is not None:
            log.info('Successfully predicted in %s', format_duration(predict_time))
            self._evaluate_predictions(meta_data=meta_data,
                                       data_split=data_split,
                                       data_type=data_type,
                                       prediction_scope=GlobalPrediction(),
                                       train_time=train_time,
                                       predict_time=predict_time,
                                       x=x,
                                       y=y,
                                       predictions=predictions,
                                       learner=learner)


class IncrementalEvaluation(Evaluation):
    """
    Repeatedly obtains and evaluates predictions from a previously trained ensemble model, e.g., a model consisting of
    several rules, using only a subset of the ensemble members with increasing size.
    """

    def __init__(self, prediction_type: PredictionType, output_writers: List[OutputWriter], min_size: int,
                 max_size: int, step_size: int):
        """
        :param min_size:    The minimum number of ensemble members to be evaluated. Must be at least 0
        :param max_size:    The maximum number of ensemble members to be evaluated. Must be greater than `min_size` or
                            0, if all ensemble members should be evaluated
        :param step_size:   The number of additional ensemble members to be considered at each repetition. Must be at
                            least 1
        """
        super().__init__(prediction_type, output_writers)
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size

    def predict_and_evaluate(self, meta_data: MetaData, data_split: DataSplit, data_type: DataType, train_time: float,
                             learner, x, y):
        if not isinstance(learner, IncrementalLearner):
            raise ValueError('Cannot obtain incremental predictions from a model of type ' + type(learner.__name__))

        incremental_predictor = self._invoke_prediction_function(learner, learner.predict_incrementally,
                                                                 learner.predict_proba_incrementally, x)

        if incremental_predictor is not None:
            step_size = self.step_size
            total_size = incremental_predictor.get_num_next()
            max_size = self.max_size

            if max_size > 0:
                total_size = min(max_size, total_size)

            min_size = self.min_size
            next_step_size = min_size if min_size > 0 else step_size
            current_size = min(next_step_size, total_size)

            while incremental_predictor.has_next():
                log.info('Predicting for %s ' + data_type.value + ' examples using a model of size %s...', x.shape[0],
                         current_size)
                start_time = timer()
                predictions = incremental_predictor.apply_next(next_step_size)
                end_time = timer()
                predict_time = end_time - start_time

                if predictions is not None:
                    log.info('Successfully predicted in %s', format_duration(predict_time))
                    self._evaluate_predictions(meta_data=meta_data,
                                               data_split=data_split,
                                               data_type=data_type,
                                               prediction_scope=IncrementalPrediction(current_size),
                                               train_time=train_time,
                                               predict_time=predict_time,
                                               x=x,
                                               y=y,
                                               predictions=predictions,
                                               learner=learner)

                next_step_size = step_size
                current_size = min(current_size + next_step_size, total_size)


class Experiment(DataSplitter.Callback):
    """
    An experiment that trains and evaluates a single multi-label classifier or ranker on a specific data set using cross
    validation or separate training and test sets.
    """

    class ExecutionHook(ABC):
        """
        An abstract base class for all operations that may be executed before or after an experiment.
        """

        @abstractmethod
        def execute(self):
            """
            Must be overridden by subclasses in order to execute the operation.
            """
            pass

    def __init__(self,
                 base_learner: BaseEstimator,
                 learner_name: str,
                 data_splitter: DataSplitter,
                 pre_training_output_writers: List[OutputWriter],
                 post_training_output_writers: List[OutputWriter],
                 pre_execution_hook: Optional[ExecutionHook] = None,
                 train_evaluation: Optional[Evaluation] = None,
                 test_evaluation: Optional[Evaluation] = None,
                 parameter_input: Optional[ParameterInput] = None,
                 persistence: Optional[ModelPersistence] = None):
        """
        :param base_learner:                    The classifier or ranker to be trained
        :param learner_name:                    The name of the classifier or ranker
        :param data_splitter:                   The method to be used for splitting the available data into training and
                                                test sets
        :param pre_training_output_writers:     A list that contains all output writers to be invoked before training
        :param post_training_output_writers:    A list that contains all output writers to be invoked after training
        :param pre_execution_hook:              An operation that should be executed before the experiment
        :param train_evaluation:                The method to be used for evaluating the predictions for the training
                                                data or None, if the predictions should not be evaluated
        :param test_evaluation:                 The method to be used for evaluating the predictions for the test data
                                                or None, if the predictions should not be evaluated
        :param parameter_input:                 The input that should be used to read the parameter settings
        :param persistence:                     The `ModelPersistence` that should be used for loading and saving models
        """
        self.base_learner = base_learner
        self.learner_name = learner_name
        self.data_splitter = data_splitter
        self.pre_training_output_writers = pre_training_output_writers
        self.post_training_output_writers = post_training_output_writers
        self.pre_execution_hook = pre_execution_hook
        self.train_evaluation = train_evaluation
        self.test_evaluation = test_evaluation
        self.parameter_input = parameter_input
        self.persistence = persistence

    def run(self):
        log.info('Starting experiment...')

        # Run pre-execution hook, if necessary...
        if self.pre_execution_hook is not None:
            self.pre_execution_hook.execute()

        self.data_splitter.run(self)

    def train_and_evaluate(self, meta_data: MetaData, data_split: DataSplit, train_x, train_y, test_x, test_y):
        base_learner = self.base_learner
        current_learner = clone(base_learner)

        # Apply parameter setting, if necessary...
        parameter_input = self.parameter_input

        if parameter_input is not None:
            params = parameter_input.read_parameters(data_split)
            current_learner.set_params(**params)
            log.info('Successfully applied parameter setting: %s', params)

        # Write output data before model is trained...
        for output_writer in self.pre_training_output_writers:
            output_writer.write_output(meta_data, train_x, train_y, data_split, current_learner)

        # Set the indices of ordinal attributes, if supported...
        if isinstance(current_learner, OrdinalAttributeLearner):
            current_learner.ordinal_attribute_indices = meta_data.get_attribute_indices({AttributeType.ORDINAL})

        # Set the indices of nominal attributes, if supported...
        if isinstance(current_learner, NominalAttributeLearner):
            current_learner.nominal_attribute_indices = meta_data.get_attribute_indices({AttributeType.NOMINAL})

        # Load model from disc, if possible, otherwise train a new model...
        loaded_learner = self.__load_model(data_split)

        if isinstance(loaded_learner, type(current_learner)):
            current_params = current_learner.get_params()
            self.__check_for_parameter_changes(expected_params=current_params,
                                               actual_params=loaded_learner.get_params())
            loaded_learner.set_params(**current_params)
            current_learner = loaded_learner
            train_time = 0
        else:
            log.info('Fitting model to %s training examples...', train_x.shape[0])
            train_time = self.__train(current_learner, train_x, train_y)
            log.info('Successfully fit model in %s', format_duration(train_time))

            # Save model to disk...
            self.__save_model(current_learner, data_split)

        # Obtain and evaluate predictions for training data, if necessary...
        evaluation = self.train_evaluation

        if evaluation is not None and data_split.is_train_test_separated():
            data_type = DataType.TRAINING
            evaluation.predict_and_evaluate(meta_data, data_split, data_type, train_time, current_learner, train_x,
                                            train_y)

        # Obtain and evaluate predictions for test data, if necessary...
        evaluation = self.test_evaluation

        if evaluation is not None:
            data_type = DataType.TEST if data_split.is_train_test_separated() else DataType.TRAINING
            evaluation.predict_and_evaluate(meta_data, data_split, data_type, train_time, current_learner, test_x,
                                            test_y)

        # Write output data after model was trained...
        for output_writer in self.post_training_output_writers:
            output_writer.write_output(meta_data, train_x, train_y, data_split, current_learner, train_time=train_time)

    @staticmethod
    def __train(learner, x, y):
        """
        Fits a learner to training data.

        :param learner: The learner
        :param x:       A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores
                        the feature values of the training examples
        :param y:       A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the
                        labels of the training examples according to the ground truth
        :return:        The time needed for training
        """
        start_time = timer()
        learner.fit(x, y)
        end_time = timer()
        return end_time - start_time

    def __load_model(self, data_split: DataSplit):
        """
        Loads the model from disk, if available.

        :param data_split:  Information about the split of the available data, the model corresponds to
        :return:            The loaded model
        """
        persistence = self.persistence

        if persistence is not None:
            return persistence.load_model(self.learner_name, data_split)

        return None

    def __save_model(self, model, data_split: DataSplit):
        """
        Saves a model to disk.

        :param model:       The model to be saved
        :param data_split:  Information about the split of the available data, the model corresponds to
        """
        persistence = self.persistence

        if persistence is not None:
            persistence.save_model(model, self.learner_name, data_split)

    @staticmethod
    def __check_for_parameter_changes(expected_params, actual_params):
        changes = []

        for key, expected_value in expected_params.items():
            expected_value = str(expected_value)
            actual_value = str(actual_params[key])

            if actual_value != expected_value:
                changes.append((key, expected_value, actual_value))

        if len(changes) > 0:
            log.warning(
                'The loaded model\'s values for the following parameters differ from the expected configuration: %s',
                reduce(
                    lambda a, b: a +
                    (', ' if len(a) > 0 else '') + '"' + b[0] + '" is "' + b[2] + '" instead of "' + b[1] + '"',
                    changes, ''))
