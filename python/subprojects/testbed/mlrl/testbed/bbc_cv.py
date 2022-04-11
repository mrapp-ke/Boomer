#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements "Bootstrap Bias Corrected Cross Validation" (BBC-CV) for evaluating different configurations of a learner and
estimating unbiased performance estimations (see https://link.springer.com/article/10.1007/s10994-018-5714-4).
"""
import logging as log
from abc import abstractmethod, ABC
from typing import List

import numpy as np
from mlrl.common.data_types import DTYPE_UINT8, DTYPE_UINT32
from mlrl.common.learners import Learner
from mlrl.testbed.data import MetaData
from mlrl.testbed.evaluation import ClassificationEvaluation, EvaluationLogOutput, EvaluationCsvOutput
from mlrl.testbed.interfaces import Randomized
from mlrl.testbed.persistence import ModelPersistence
from mlrl.testbed.training import CrossValidation, DataSet
from sklearn.base import clone
from sklearn.utils import check_random_state


class BbcCvAdapter(CrossValidation):
    """
    An adapter that must be implemented for each type of model to be used with BBC-CV to obtain predictions for given
    test examples.
    """

    def __init__(self, data_set: DataSet, num_folds: int, model_dir: str):
        """
        :param model_dir: The path of the directory where the models are stored
        """
        super().__init__(data_set, num_folds, -1)
        self.persistence = ModelPersistence(model_dir=model_dir)
        self.learner = None
        self.configuration = None
        self.store_true_labels = True
        self.require_dense = [True, True]
        self.predictions = []
        self.configurations = []
        self.true_labels = None
        self.meta_data = None

    def _train_and_evaluate(self, meta_data: MetaData, train_indices, train_x, train_y, test_indices, test_x, test_y,
                            first_fold: int, current_fold: int, last_fold: int, num_folds: int):
        self.meta_data = meta_data
        num_total_examples = test_x.shape[0] + (0 if test_indices is None else train_x.shape[0])
        num_labels = test_y.shape[1] if len(test_y.shape) > 1 else 1

        # Update true labels, if necessary...
        if self.store_true_labels:
            true_labels = self.true_labels

            if true_labels is None:
                if test_indices is None:
                    true_labels = test_y
                else:
                    true_labels = np.empty((num_total_examples, num_labels), dtype=DTYPE_UINT8)

                self.true_labels = true_labels

            if test_indices is not None:
                true_labels[test_indices] = test_y

        # Load theory...
        current_learner = clone(self.learner)
        current_learner.set_params(**self.configuration)
        model_name = current_learner.get_name()
        model = self.persistence.load_model(model_name=model_name, fold=current_fold, raise_exception=True)

        predictions = self.predictions
        configurations = self.configurations
        self._store_predictions(model, test_indices, test_x, train_y, num_total_examples, num_labels, predictions,
                                configurations, current_fold, last_fold, num_folds)

    def run(self):
        self.predictions = []
        self.configurations = []
        self.true_labels = None
        self.meta_data = None
        super().run()

    @abstractmethod
    def _store_predictions(self, model, test_indices, test_x, train_y, num_total_examples: int, num_labels: int,
                           predictions, configurations, current_fold: int, last_fold: int, num_folds: int):
        """
        Must be implemented by subclasses to store the predictions provided by a specific model for the given test
        examples. The predictions, together with the corresponding configuration, must be stored in the given lists
        `predictions` and `configurations`. It is possible to evaluate more than one configurations by modifying the
        given model accordingly.

        :param model:               The model that should be used to make predictions
        :param test_indices:        The indices of the test examples
        :param test_x:              An array of type `float`, shape `(num_examples, num_features)`, representing the
                                    features of the test examples
        :param num_total_examples:  The total number of examples
        :param num_labels:          The number of labels
        :param predictions:         The list that should be used to store predictions
        :param configurations:      The list that should be used to store configurations
        :param current_fold:        The current fold starting at 0, or 0 if no cross validation is used
        :param last_fold:           The last fold or 0, if no cross validation is used
        :param num_folds:           The total number of cross validation folds or 1, if no cross validation is used
        """
        pass

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass


class BbcCvObserver(ABC):
    """
    A base class for all observers that should be notified about the predictions and ground truth labellings that result
    from applying the BBC-CV method.
    """

    @abstractmethod
    def evaluate(self, configurations: List[dict], meta_data: MetaData, ground_truth_tuning: np.ndarray,
                 predictions_tuning: np.ndarray, ground_truth_test: np.ndarray, predictions_test: np.ndarray,
                 current_bootstrap: int, num_bootstraps: int):
        """
        :param configurations:      The configurations that have been provided to the BBC-CV method
        :param meta_data:           The meta data of the data set
        :param ground_truth_tuning: The ground truth of the examples that belong to the tuning set
        :param predictions_tuning:  The predictions for the examples that belong to the tuning set
        :param ground_truth_test:   The ground truth of the examples that belong to the test set
        :param predictions_test:    The predictions for the examples that belong to the test set
        :param current_bootstrap:   The current bootstrap iteration
        :param num_bootstraps:      The total number of bootstrap iterations
        """
        pass


class Bootstrapping(Randomized):

    @abstractmethod
    def bootstrap(self, meta_data: MetaData, prediction_matrix, ground_truth_matrix, configurations: List[dict],
                  observer: BbcCvObserver):
        pass


class DefaultBbcCvObserver(BbcCvObserver):
    """
    An observer that determines the best configuration per bootstrap iteration and computes the evaluation measures
    averaged over all iterations.
    """

    def __init__(self, target_measure, target_measure_is_loss: bool, output_dir: str = None):
        """
        :param target_measure:          The target measure to be used for parameter tuning
        :param target_measure_is_loss:  True, if the target measure is a loss, False otherwise
        :param output_dir:              The path of the directory where the evaluation results should be stored
        """
        self.target_measure = target_measure
        self.target_measure_is_loss = target_measure_is_loss
        evaluation_outputs = [EvaluationLogOutput(output_individual_folds=False)]

        if output_dir is not None:
            evaluation_outputs.append(EvaluationCsvOutput(output_dir=output_dir, output_individual_folds=False))

        self.evaluation = ClassificationEvaluation(*evaluation_outputs)

    def evaluate(self, configurations: List[dict], meta_data: MetaData, ground_truth_tuning: np.ndarray,
                 predictions_tuning: np.ndarray, ground_truth_test: np.ndarray, predictions_test: np.ndarray,
                 current_bootstrap: int, num_bootstraps: int):
        target_measure = self.target_measure
        target_measure_is_loss = self.target_measure_is_loss
        num_configurations = len(configurations)
        evaluation_scores_tuning = np.empty(num_configurations, dtype=float)

        for k in range(num_configurations):
            predictions = predictions_tuning[:, k, :]
            evaluation_scores_tuning[k] = target_measure(ground_truth_tuning, predictions)

        best_k = np.argmin(evaluation_scores_tuning) if target_measure_is_loss else np.argmax(evaluation_scores_tuning)
        best_predictions = predictions_test[:, best_k, :]
        self.evaluation.evaluate('best_configuration', meta_data, best_predictions, ground_truth_test, first_fold=0,
                                 current_fold=current_bootstrap, last_fold=num_bootstraps - 1, num_folds=num_bootstraps,
                                 train_time=0, predict_time=0)


class BbcCv(Randomized):
    """
    An implementation of "Bootstrap Bias Corrected Cross Validation" (BBC-CV).
    """

    def __init__(self, configurations: List[dict], adapter: BbcCvAdapter, bootstrapping: Bootstrapping,
                 learner: Learner):
        """
        :param configurations:  A list that contains the configurations to be evaluated
        :param adapter:         The `BbcCvAdapter` to be used
        :param learner:         The learner to be evaluated
        """
        super().__init__()
        self.configurations = configurations
        self.adapter = adapter
        self.bootstrapping = bootstrapping
        self.learner = learner
        self.prediction_matrix_ = None
        self.ground_truth_matrix_ = None
        self.configurations_ = None
        self.meta_data = None

    def store_predictions(self):
        configurations = self.configurations
        num_configurations = len(configurations)
        log.info('%s configurations have been specified...', num_configurations)

        # Store predictions of the different models...
        random_state = self.random_state
        adapter = self.adapter
        adapter.random_state = random_state
        adapter.learner = self.learner
        list_of_predictions: List[np.ndarray] = []
        list_of_configurations: List[dict] = []
        ground_truth_matrix = None

        for index, config in enumerate(configurations):
            log.info('Storing predictions of configuration %s / %s...', str(index + 1), num_configurations)

            adapter.configuration = config
            adapter.store_true_labels = ground_truth_matrix is None

            try:
                adapter.run()

                list_of_predictions.extend(adapter.predictions)
                list_of_configurations.extend(adapter.configurations)

                if ground_truth_matrix is None:
                    ground_truth_matrix = adapter.true_labels
            except (FileNotFoundError, ArithmeticError):
                # Ignore configuration if a model file is missing...
                pass

        self.meta_data = adapter.meta_data

        # Create 3-dimensional prediction matrix....
        prediction_matrix = np.moveaxis(np.dstack(list_of_predictions), source=1, destination=2)
        prediction_matrix = np.where(prediction_matrix > 0, 1, 0)
        self.prediction_matrix_ = prediction_matrix
        self.ground_truth_matrix_ = ground_truth_matrix
        self.configurations_ = list_of_configurations

    def evaluate(self, observer: BbcCvObserver):
        """
        :param observer: The `BbcCvObserver` to be used
        """
        prediction_matrix = self.prediction_matrix_
        ground_truth_matrix = self.ground_truth_matrix_
        configurations = self.configurations_
        random_state = self.random_state
        meta_data = self.meta_data

        # Bootstrap sampling...
        bootstrapping = self.bootstrapping
        bootstrapping.random_state = random_state
        bootstrapping.bootstrap(meta_data, prediction_matrix, ground_truth_matrix, configurations, observer)


class CV(CrossValidation):

    def __init__(self, data_set: DataSet, num_folds: int, prediction_matrix, ground_truth_matrix,
                 configurations: List[dict], observer: BbcCvObserver):
        super().__init__(data_set, num_folds, -1)
        self.prediction_matrix = prediction_matrix
        self.ground_truth_matrix = ground_truth_matrix
        self.configurations = configurations
        self.observer = observer

    def _train_and_evaluate(self, meta_data: MetaData, train_indices, train_x, train_y, test_indices, test_x, test_y,
                            first_fold: int, current_fold: int, last_fold: int, num_folds: int):
        configurations = self.configurations
        prediction_matrix = self.prediction_matrix
        ground_truth_matrix = self.ground_truth_matrix
        observer = self.observer
        ground_truth_tuning = ground_truth_matrix[train_indices, :]
        predictions_tuning = prediction_matrix[train_indices, :, :]
        ground_truth_test = ground_truth_matrix[test_indices, :]
        predictions_test = prediction_matrix[test_indices, :, :]
        observer.evaluate(configurations, meta_data, ground_truth_tuning, predictions_tuning, ground_truth_test,
                          predictions_test, current_fold, num_folds)


class CVBootstrapping(Bootstrapping):

    def __init__(self, data_set: DataSet, num_folds: int):
        self.data_set = data_set
        self.num_folds = num_folds

    def bootstrap(self, meta_data: MetaData, prediction_matrix, ground_truth_matrix, configurations: List[dict],
                  observer: BbcCvObserver):
        num_configurations = prediction_matrix.shape[1]
        log.info('%s configurations have been evaluated...', num_configurations)
        cv = CV(self.data_set, self.num_folds, prediction_matrix, ground_truth_matrix, configurations, observer)
        cv.random_state = self.random_state
        cv.run()


class DefaultBootstrapping(Bootstrapping):

    def __init__(self, num_bootstraps: int):
        self.num_bootstraps = num_bootstraps

    def bootstrap(self, meta_data: MetaData, prediction_matrix, ground_truth_matrix, configurations: List[dict],
                  observer: BbcCvObserver):
        num_bootstraps = self.num_bootstraps
        num_examples = prediction_matrix.shape[0]
        num_configurations = prediction_matrix.shape[1]
        log.info('%s configurations have been evaluated...', num_configurations)
        bootstrapped_indices = np.empty(num_examples, dtype=DTYPE_UINT32)
        mask_test = np.empty(num_examples, dtype=np.bool)
        rng = check_random_state(self.random_state)
        rng_randint = rng.randint

        for i in range(num_bootstraps):
            mask_test[:] = True
            log.info('Sampling bootstrap examples %s / %s...', (i + 1), num_bootstraps)

            for j in range(num_examples):
                index = rng_randint(num_examples)
                bootstrapped_indices[j] = index
                mask_test[index] = False

            ground_truth_tuning = ground_truth_matrix[bootstrapped_indices, :]
            predictions_tuning = prediction_matrix[bootstrapped_indices, :, :]
            ground_truth_test = ground_truth_matrix[mask_test, :]
            predictions_test = prediction_matrix[mask_test, :, :]
            observer.evaluate(configurations, meta_data, ground_truth_tuning, predictions_tuning, ground_truth_test,
                              predictions_test, i, num_bootstraps)
