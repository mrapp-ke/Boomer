#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for evaluating the predictions or rankings provided by a multi-label learner according to different
measures. The evaluation results can be written to one or several outputs, e.g. to the console or to a file.
"""
import logging as log
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Optional

import numpy as np
import sklearn.metrics as metrics
from mlrl.common.arrays import enforce_dense
from mlrl.common.data_types import DTYPE_UINT8
from mlrl.testbed.data import MetaData, save_arff_file, Label
from mlrl.testbed.io import open_writable_csv_file, create_csv_dict_writer, clear_directory, SUFFIX_ARFF, \
    get_file_name_per_fold
from sklearn.utils.multiclass import is_multilabel

# The name of the accuracy metric
ACCURACY = 'Acc.'

# The name of the 0/1 loss metric.
ZERO_ONE_LOSS = '0/1 Loss'

# The name of the precision metric
PRECISION = 'Prec.'

# The name of the recall metric
RECALL = 'Rec.'

# The name of the F1 metric
F1 = 'F1'

# The name of the hamming loss metric
HAMMING_LOSS = 'Hamm. Loss'

# The name of the hamming accuracy metric
HAMMING_ACCURACY = 'Hamm. Acc.'

# The name of the subset 0/1 loss metric
SUBSET_ZERO_ONE_LOSS = 'Subs. 0/1 Loss'

# The name of the subset accuracy metric
SUBSET_ACCURACY = 'Subs. Acc.'

# The name of the micro-averaged precision metric
MICRO_PRECISION = 'Mi. Prec.'

# The name of the macro-averaged precision metric
MACRO_PRECISION = 'Ma. Prec.'

# The name of the micro-averaged recall metric
MICRO_RECALL = 'Mi. Rec.'

# The name of the macro-averaged recall metric
MACRO_RECALL = 'Ma. Rec.'

# The name of the micro-averaged F1 metric
MICRO_F1 = 'Mi. F1'

# The name of the macro-averaged F1 metric
MACRO_F1 = 'Ma. F1'

# The name of the example-based precision metric
EX_BASED_PRECISION = 'Ex.-based Prec.'

# The name of the example-based recall metric
EX_BASED_RECALL = 'Ex.-based Rec.'

# The name of the example-based F1 metric
EX_BASED_F1 = 'Ex.-based F1'

# The name of the rank loss metric
RANK_LOSS = 'Rank Loss'

# The time needed to train the model
TIME_TRAIN = 'Training Time'

# The time needed to make predictions
TIME_PREDICT = 'Prediction Time'


class Evaluation(ABC):
    """
    An abstract base class for all classes that evaluate the predictions provided by a classifier or ranker.
    """

    @abstractmethod
    def evaluate(self, experiment_name: str, meta_data: MetaData, predictions, ground_truth, first_fold: int,
                 current_fold: int, last_fold: int, num_folds: int, train_time: float, predict_time: float):
        """
        Evaluates the predictions provided by a classifier or ranker.

        :param experiment_name: The name of the experiment
        :param meta_data:       The meta data of the data set
        :param predictions:     The predictions provided by the classifier
        :param ground_truth:    The true labels
        :param first_fold:      The first cross validation fold or 0, if no cross validation is used
        :param current_fold:    The current cross validation fold starting at 0, or 0 if no cross validation is used
        :param last_fold:       The last cross validation fold or 0, if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        :param train_time:      The time needed to train the model
        :param predict_time:    The time needed to make predictions
        """
        pass


class EvaluationResult:
    """
    Stores the evaluation results according to different measures.
    """

    def __init__(self):
        self.measures: Set[str] = set()
        self.results: Optional[List[Dict[str, float]]] = None

    def put(self, name: str, score: float, fold: int, num_folds: int):
        """
        Adds a new score according to a specific measure to the evaluation result.

        :param name:        The name of the measure
        :param score:       The score according to the measure
        :param fold:        The fold the score corresponds to
        :param num_folds:   The total number of cross validation folds
        """

        if self.results is None:
            self.results = [{} for _ in range(num_folds)]
        elif len(self.results) != num_folds:
            raise AssertionError('Inconsistent number of total folds given')

        self.measures.add(name)
        values = self.results[fold]
        values[name] = score

    def get(self, name: str, fold: int) -> float:
        """
        Returns the score according to a specific measure and fold.

        :param name:    The name of the measure
        :param fold:    The fold the score corresponds to
        :return:        The score
        """

        values = self.results[fold] if self.results is not None else None

        if values is None:
            raise AssertionError('No evaluation results available')

        return values[name]

    def dict(self, fold: int) -> Dict:
        values = self.results[fold].copy() if self.results is not None else None

        if values is None:
            raise AssertionError('No evaluation results available')

        return values

    def avg(self, name: str) -> (float, float):
        """
        Returns the score and standard deviation according to a specific measure averaged over all available folds.

        :param name:    The name of the measure
        :return:        A tuple consisting of the averaged score and standard deviation
        """
        values = []

        for i in range(len(self.results)):
            if len(self.results[i]) > 0:
                values.append(self.get(name, i))

        values = np.array(values)
        return np.average(values), np.std(values)

    def avg_dict(self) -> Dict:
        result: Dict[str, float] = {}

        for measure in self.measures:
            score, std_dev = self.avg(measure)
            result[measure] = score
            result['Std.-dev. ' + measure] = std_dev.item()

        return result


class EvaluationOutput(ABC):
    """
    An abstract base class for all outputs, evaluation results may be written to.
    """

    def __init__(self, output_predictions: bool, output_individual_folds: bool):
        """
        :param output_predictions:      True, if predictions provided by a classifier or ranker should be written to the
                                        output, False otherwise
        :param output_individual_folds: True, if the evaluation results for individual cross validation folds should be
                                        written to the outputs, False, if only the overall evaluation results, i.e.,
                                        averaged over all folds, should be written to the outputs
        """
        self.output_predictions = output_predictions
        self.output_individual_folds = output_individual_folds

    @abstractmethod
    def write_evaluation_results(self, experiment_name: str, evaluation_result: EvaluationResult, total_folds: int,
                                 fold: int = None):
        """
        Writes an evaluation result to the output.

        :param experiment_name:     The name of the experiment
        :param evaluation_result:   The evaluation result to be written
        :param total_folds:         The total number of folds
        :param fold:                The fold for which the results should be written or None, if no cross validation is
                                    used or if the overall results, averaged over all folds, should be written
        """
        pass

    @abstractmethod
    def write_predictions(self, experiment_name: str, meta_data: MetaData, predictions, ground_truth, total_folds: int,
                          fold: int = None):
        """
        Writes predictions to the output.

        :param experiment_name: The name of the experiment
        :param meta_data:       The meta data of the data set
        :param predictions:     The predictions
        :param ground_truth:    The ground truth
        :param total_folds:     The total number of folds
        :param fold:            The fold for which the predictions should be written or None, if no cross validation is
                                used
        """
        pass


class EvaluationLogOutput(EvaluationOutput):
    """
    Outputs evaluation result using the logger.
    """

    def __init__(self, output_predictions: bool = False, output_individual_folds: bool = True):
        super().__init__(output_predictions, output_individual_folds)

    def write_evaluation_results(self, experiment_name: str, evaluation_result: EvaluationResult, total_folds: int,
                                 fold: int = None):
        if fold is None or self.output_individual_folds:
            text = ''

            for measure in sorted(evaluation_result.measures):
                if measure != TIME_TRAIN and measure != TIME_PREDICT:
                    if len(text) > 0:
                        text += '\n'

                    if fold is None:
                        score, std_dev = evaluation_result.avg(measure)
                        text += (measure + ': ' + str(score))

                        if total_folds > 1:
                            text += (' ±' + str(std_dev))
                    else:
                        score = evaluation_result.get(measure, fold)
                        text += (measure + ': ' + str(score))

            msg = ('Overall evaluation result for experiment \"' + experiment_name + '\"' if fold is None else
                   'Evaluation result for experiment \"' + experiment_name + '\" (Fold ' + str(
                       fold + 1) + ')') + ':\n\n%s\n'
            log.info(msg, text)

    def write_predictions(self, experiment_name: str, meta_data: MetaData, predictions, ground_truth, total_folds: int,
                          fold: int = None):
        if self.output_predictions:
            text = 'Ground truth:\n\n' + np.array2string(ground_truth) + '\n\nPredictions:\n\n' + np.array2string(
                predictions)
            msg = ('Predictions for experiment \"' + experiment_name + '\"' if fold is None else
                   'Predictions for experiment \"' + experiment_name + '\" (Fold ' + str(fold + 1) + ')') + ':\n\n%s\n'
            log.info(msg, text)


class EvaluationCsvOutput(EvaluationOutput):
    """
    Writes evaluation results to CSV files.
    """

    def __init__(self, output_dir: str, clear_dir: bool = True, output_predictions: bool = False,
                 output_individual_folds: bool = True):
        """
        :param output_dir:  The path of the directory, the CSV files should be written to
        :param clear_dir:   True, if the directory, the CSV files should be written to, should be cleared
        """
        super().__init__(output_predictions, output_individual_folds)
        self.output_dir = output_dir
        self.clear_dir = clear_dir

    def write_evaluation_results(self, experiment_name: str, evaluation_result: EvaluationResult, total_folds: int,
                                 fold: int = None):
        if fold is None or self.output_individual_folds:
            self.__clear_dir_if_necessary()
            columns = evaluation_result.avg_dict() if fold is None else evaluation_result.dict(fold)
            header = sorted(columns.keys())
            header.insert(0, 'Approach')
            columns['Approach'] = experiment_name

            with open_writable_csv_file(self.output_dir, 'evaluation', fold, append=True) as csv_file:
                csv_writer = create_csv_dict_writer(csv_file, header)
                csv_writer.writerow(columns)

    def write_predictions(self, experiment_name: str, meta_data: MetaData, predictions, ground_truth, total_folds: int,
                          fold: int = None):
        if self.output_predictions:
            self.__clear_dir_if_necessary()
            file_name = get_file_name_per_fold('predictions_' + experiment_name, SUFFIX_ARFF, fold)
            attributes = [Label('Ground Truth ' + label.attribute_name) for label in meta_data.labels]
            labels = [Label('Prediction ' + label.attribute_name) for label in meta_data.labels]
            prediction_meta_data = MetaData(attributes, labels, labels_at_start=False)
            save_arff_file(self.output_dir, file_name, ground_truth, predictions, prediction_meta_data)

    def __clear_dir_if_necessary(self):
        """
        Clears the output directory, if necessary.
        """
        if self.clear_dir:
            clear_directory(self.output_dir)
            self.clear_dir = False


class AbstractEvaluation(Evaluation):
    """
    An abstract base class for all classes that evaluate the predictions provided by a classifier or ranker and allow to
    write the results to one or several outputs.
    """

    def __init__(self, *args: EvaluationOutput):
        """
        :param args: The outputs, the evaluation results should be written to
        """
        self.outputs = args
        self.results: Dict[str, EvaluationResult] = {}

    def evaluate(self, experiment_name: str, meta_data: MetaData, predictions, ground_truth, first_fold: int,
                 current_fold: int, last_fold: int, num_folds: int, train_time: float, predict_time: float):
        result = self.results[experiment_name] if experiment_name in self.results else EvaluationResult()
        self.results[experiment_name] = result
        result.put(TIME_TRAIN, train_time, current_fold, num_folds)
        result.put(TIME_PREDICT, predict_time, current_fold, num_folds)
        self._populate_result(result, predictions, ground_truth, current_fold=current_fold, num_folds=num_folds)
        self.__write_predictions(experiment_name, meta_data, predictions, ground_truth, current_fold=current_fold,
                                 num_folds=num_folds)
        self.__write_evaluation_result(experiment_name, result, first_fold=first_fold, current_fold=current_fold,
                                       last_fold=last_fold, num_folds=num_folds)

    @abstractmethod
    def _populate_result(self, result: EvaluationResult, predictions, ground_truth, current_fold: int, num_folds: int):
        pass

    def __write_predictions(self, experiment_name: str, meta_data: MetaData, predictions, ground_truth,
                            current_fold: int, num_folds: int):
        """
        Writes predictions to the outputs.

        :param experiment_name: The name of the experiment
        :param meta_data:       The meta data of the data set
        :param predictions:     The predictions
        :param ground_truth:    The ground truth
        :param current_fold:    The current cross validation fold or 0, if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        """

        for output in self.outputs:
            output.write_predictions(experiment_name, meta_data, predictions, ground_truth, num_folds,
                                     current_fold if num_folds > 1 else None)

    def __write_evaluation_result(self, experiment_name: str, result: EvaluationResult, first_fold: int,
                                  current_fold: int, last_fold: int, num_folds: int):
        """
        Writes an evaluation result to the outputs.

        :param experiment_name: The name of the experiment
        :param result:          The evaluation result
        :param first_fold:      The first cross validation fold or 0, if no cross validation is used
        :param current_fold:    The current cross validation fold or 0, if no cross validation is used
        :param last_fold        The last cross validation fold or 0, if no cross validation is used
        :param num_folds:       The total number of cross validation folds or 1, if no cross validation is used
        """

        if num_folds > 1:
            for output in self.outputs:
                output.write_evaluation_results(experiment_name, result, num_folds, current_fold)

        if num_folds == 1 or (current_fold == last_fold and abs(last_fold - first_fold) > 0):
            for output in self.outputs:
                output.write_evaluation_results(experiment_name, result, num_folds)


class ClassificationEvaluation(AbstractEvaluation):
    """
    Evaluates the predictions of a single- or multi-label classifier according to commonly used bipartition measures.
    """

    def __init__(self, *args: EvaluationOutput):
        super().__init__(*args)

    def _populate_result(self, result: EvaluationResult, predictions, ground_truth, current_fold: int, num_folds: int):
        if is_multilabel(ground_truth):
            hamming_loss = metrics.hamming_loss(ground_truth, predictions)
            result.put(HAMMING_LOSS, hamming_loss, current_fold, num_folds)
            result.put(HAMMING_ACCURACY, 1 - hamming_loss, current_fold, num_folds)
            subset_accuracy = metrics.accuracy_score(ground_truth, predictions)
            result.put(SUBSET_ACCURACY, subset_accuracy, current_fold, num_folds)
            result.put(SUBSET_ZERO_ONE_LOSS, 1 - subset_accuracy, current_fold, num_folds)
            result.put(MICRO_PRECISION, metrics.precision_score(ground_truth, predictions, average='micro',
                                                                zero_division=1), current_fold, num_folds)
            result.put(MICRO_RECALL, metrics.recall_score(ground_truth, predictions, average='micro', zero_division=1),
                       current_fold, num_folds)
            result.put(MICRO_F1, metrics.f1_score(ground_truth, predictions, average='micro', zero_division=1),
                       current_fold, num_folds)
            result.put(MACRO_PRECISION, metrics.precision_score(ground_truth, predictions, average='macro',
                                                                zero_division=1), current_fold, num_folds)
            result.put(MACRO_RECALL, metrics.recall_score(ground_truth, predictions, average='macro', zero_division=1),
                       current_fold, num_folds)
            result.put(MACRO_F1, metrics.f1_score(ground_truth, predictions, average='macro', zero_division=1),
                       current_fold, num_folds)
            result.put(EX_BASED_PRECISION, metrics.precision_score(ground_truth, predictions, average='samples',
                                                                   zero_division=1), current_fold, num_folds)
            result.put(EX_BASED_RECALL, metrics.recall_score(ground_truth, predictions, average='samples',
                                                             zero_division=1), current_fold, num_folds)
            result.put(EX_BASED_F1, metrics.f1_score(ground_truth, predictions, average='samples', zero_division=1),
                       current_fold, num_folds)
        else:
            predictions = np.ravel(enforce_dense(predictions, order='C', dtype=DTYPE_UINT8))
            ground_truth = np.ravel(enforce_dense(ground_truth, order='C', dtype=DTYPE_UINT8))
            accuracy = metrics.accuracy_score(ground_truth, predictions)
            result.put(ACCURACY, accuracy, current_fold, num_folds)
            result.put(ZERO_ONE_LOSS, 1 - accuracy, current_fold, num_folds)
            result.put(PRECISION, metrics.precision_score(ground_truth, predictions, zero_division=1), current_fold,
                       num_folds)
            result.put(RECALL, metrics.recall_score(ground_truth, predictions, zero_division=1), current_fold,
                       num_folds)
            result.put(F1, metrics.f1_score(ground_truth, predictions, zero_division=1), current_fold, num_folds)


class RankingEvaluation(AbstractEvaluation):
    """
    Evaluates the predictions of a multi-label ranker according to commonly used ranking measures.
    """

    def __init__(self, *args: EvaluationOutput):
        super().__init__(*args)

    def _populate_result(self, result: EvaluationResult, predictions, ground_truth, current_fold: int, num_folds: int):
        result.put(RANK_LOSS, metrics.label_ranking_loss(ground_truth, predictions), current_fold, num_folds)
