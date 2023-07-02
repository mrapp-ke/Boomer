"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for evaluating the predictions or rankings provided by a multi-label learner according to different
measures. The evaluation results can be written to one or several outputs, e.g., to the console or to a file.
"""
from abc import ABC
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import sklearn.metrics as metrics

from sklearn.utils.multiclass import is_multilabel

from mlrl.common.arrays import enforce_dense
from mlrl.common.data_types import DTYPE_UINT8
from mlrl.common.options import Options

from mlrl.testbed.data import MetaData
from mlrl.testbed.data_splitting import CrossValidationOverall, DataSplit, DataType
from mlrl.testbed.format import OPTION_DECIMALS, OPTION_PERCENTAGE, Formatter, filter_formatters, format_table
from mlrl.testbed.output_writer import Formattable, OutputWriter, Tabularizable
from mlrl.testbed.prediction_scope import PredictionScope, PredictionType

OPTION_ENABLE_ALL = 'enable_all'

OPTION_HAMMING_LOSS = 'hamming_loss'

OPTION_HAMMING_ACCURACY = 'hamming_accuracy'

OPTION_SUBSET_ZERO_ONE_LOSS = 'subset_zero_one_loss'

OPTION_SUBSET_ACCURACY = 'subset_accuracy'

OPTION_MICRO_PRECISION = 'micro_precision'

OPTION_MICRO_RECALL = 'micro_recall'

OPTION_MICRO_F1 = 'micro_f1'

OPTION_MICRO_JACCARD = 'micro_jaccard'

OPTION_MACRO_PRECISION = 'macro_precision'

OPTION_MACRO_RECALL = 'macro_recall'

OPTION_MACRO_F1 = 'macro_f1'

OPTION_MACRO_JACCARD = 'macro_jaccard'

OPTION_EXAMPLE_WISE_PRECISION = 'example_wise_precision'

OPTION_EXAMPLE_WISE_RECALL = 'example_wise_recall'

OPTION_EXAMPLE_WISE_F1 = 'example_wise_f1'

OPTION_EXAMPLE_WISE_JACCARD = 'example_wise_jaccard'

OPTION_ACCURACY = 'accuracy'

OPTION_ZERO_ONE_LOSS = 'zero_one_loss'

OPTION_PRECISION = 'precision'

OPTION_RECALL = 'recall'

OPTION_F1 = 'f1'

OPTION_JACCARD = 'jaccard'

OPTION_MEAN_ABSOLUTE_ERROR = 'mean_absolute_error'

OPTION_MEAN_SQUARED_ERROR = 'mean_squared_error'

OPTION_MEDIAN_ABSOLUTE_ERROR = 'mean_absolute_error'

OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR = 'mean_absolute_percentage_error'

OPTION_RANK_LOSS = 'rank_loss'

OPTION_COVERAGE_ERROR = 'coverage_error'

OPTION_LABEL_RANKING_AVERAGE_PRECISION = 'lrap'

OPTION_DISCOUNTED_CUMULATIVE_GAIN = 'dcg'

OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN = 'ndcg'

OPTION_TRAINING_TIME = 'training_time'

OPTION_PREDICTION_TIME = 'prediction_time'


class EvaluationFunction(Formatter):
    """
    An evaluation function.
    """

    def __init__(self, option: str, name: str, evaluation_function, percentage: bool = True, **kwargs):
        """
        :param evaluation_function: The function that should be invoked for evaluation
        """
        super().__init__(option, name, percentage)
        self.evaluation_function = evaluation_function
        self.kwargs = kwargs

    def evaluate(self, ground_truth, predictions) -> float:
        """
        Applies the evaluation function to given predictions and ground truth labels.

        :param ground_truth:    The ground truth
        :param predictions:     The predictions
        :return:                An evaluation score
        """
        return self.evaluation_function(ground_truth, predictions, **self.kwargs)


ARGS_SINGLE_LABEL = {'zero_division': 1}

ARGS_MICRO = {'average': 'micro', 'zero_division': 1}

ARGS_MACRO = {'average': 'macro', 'zero_division': 1}

ARGS_EXAMPLE_WISE = {'average': 'samples', 'zero_division': 1}

EVALUATION_MEASURE_TRAINING_TIME = Formatter(OPTION_TRAINING_TIME, 'Training Time')

EVALUATION_MEASURE_PREDICTION_TIME = Formatter(OPTION_PREDICTION_TIME, 'Prediction Time')

MULTI_LABEL_EVALUATION_MEASURES: List[Formatter] = [
    EvaluationFunction(OPTION_HAMMING_ACCURACY, 'Hamming Accuracy', lambda a, b: 1 - metrics.hamming_loss(a, b)),
    EvaluationFunction(OPTION_HAMMING_LOSS, 'Hamming Loss', metrics.hamming_loss),
    EvaluationFunction(OPTION_SUBSET_ACCURACY, 'Subset Accuracy', metrics.accuracy_score),
    EvaluationFunction(OPTION_SUBSET_ZERO_ONE_LOSS, 'Subset 0/1 Loss', lambda a, b: 1 - metrics.accuracy_score(a, b)),
    EvaluationFunction(OPTION_MICRO_PRECISION, 'Micro Precision', metrics.precision_score, **ARGS_MICRO),
    EvaluationFunction(OPTION_MICRO_RECALL, 'Micro Recall', metrics.recall_score, **ARGS_MICRO),
    EvaluationFunction(OPTION_MICRO_F1, 'Micro F1', metrics.f1_score, **ARGS_MICRO),
    EvaluationFunction(OPTION_MICRO_JACCARD, 'Micro Jaccard', metrics.jaccard_score, **ARGS_MICRO),
    EvaluationFunction(OPTION_MACRO_PRECISION, 'Macro Precision', metrics.precision_score, **ARGS_MACRO),
    EvaluationFunction(OPTION_MACRO_RECALL, 'Macro Recall', metrics.recall_score, **ARGS_MACRO),
    EvaluationFunction(OPTION_MACRO_F1, 'Macro F1', metrics.f1_score, **ARGS_MACRO),
    EvaluationFunction(OPTION_MACRO_JACCARD, 'Macro Jaccard', metrics.jaccard_score, **ARGS_MACRO),
    EvaluationFunction(OPTION_EXAMPLE_WISE_PRECISION, 'Example-wise Precision', metrics.precision_score,
                       **ARGS_EXAMPLE_WISE),
    EvaluationFunction(OPTION_EXAMPLE_WISE_RECALL, 'Example-wise Recall', metrics.recall_score, **ARGS_EXAMPLE_WISE),
    EvaluationFunction(OPTION_EXAMPLE_WISE_F1, 'Example-wise F1', metrics.f1_score, **ARGS_EXAMPLE_WISE),
    EvaluationFunction(OPTION_EXAMPLE_WISE_JACCARD, 'Example-wise Jaccard', metrics.jaccard_score, **ARGS_EXAMPLE_WISE),
    EVALUATION_MEASURE_TRAINING_TIME,
    EVALUATION_MEASURE_PREDICTION_TIME,
]

SINGLE_LABEL_EVALUATION_MEASURES: List[Formatter] = [
    EvaluationFunction(OPTION_ACCURACY, 'Accuracy', metrics.accuracy_score),
    EvaluationFunction(OPTION_ZERO_ONE_LOSS, '0/1 Loss', lambda a, b: 1 - metrics.accuracy_score(a, b)),
    EvaluationFunction(OPTION_PRECISION, 'Precision', metrics.precision_score, **ARGS_SINGLE_LABEL),
    EvaluationFunction(OPTION_RECALL, 'Recall', metrics.recall_score, **ARGS_SINGLE_LABEL),
    EvaluationFunction(OPTION_F1, 'F1', metrics.f1_score, **ARGS_SINGLE_LABEL),
    EvaluationFunction(OPTION_JACCARD, 'Jaccard', metrics.jaccard_score, **ARGS_SINGLE_LABEL),
    EVALUATION_MEASURE_TRAINING_TIME,
    EVALUATION_MEASURE_PREDICTION_TIME,
]

REGRESSION_EVALUATION_MEASURES: List[Formatter] = [
    EvaluationFunction(OPTION_MEAN_ABSOLUTE_ERROR, 'Mean Absolute Error', metrics.mean_absolute_error,
                       percentage=False),
    EvaluationFunction(OPTION_MEAN_SQUARED_ERROR, 'Mean Squared Error', metrics.mean_squared_error, percentage=False),
    EvaluationFunction(OPTION_MEDIAN_ABSOLUTE_ERROR,
                       'Median Absolute Error',
                       metrics.median_absolute_error,
                       percentage=False),
    EvaluationFunction(OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR,
                       'Mean Absolute Percentage Error',
                       metrics.mean_absolute_percentage_error,
                       percentage=False),
    EVALUATION_MEASURE_TRAINING_TIME,
    EVALUATION_MEASURE_PREDICTION_TIME,
]

RANKING_EVALUATION_MEASURES: List[Formatter] = [
    EvaluationFunction(OPTION_RANK_LOSS, 'Ranking Loss', metrics.label_ranking_loss, percentage=False),
    EvaluationFunction(OPTION_COVERAGE_ERROR, 'Coverage Error', metrics.coverage_error, percentage=False),
    EvaluationFunction(OPTION_LABEL_RANKING_AVERAGE_PRECISION,
                       'Label Ranking Average Precision',
                       metrics.label_ranking_average_precision_score,
                       percentage=False),
    EvaluationFunction(OPTION_DISCOUNTED_CUMULATIVE_GAIN,
                       'Discounted Cumulative Gain',
                       metrics.dcg_score,
                       percentage=False),
    EvaluationFunction(OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, 'NDCG', metrics.ndcg_score),
    EVALUATION_MEASURE_TRAINING_TIME,
    EVALUATION_MEASURE_PREDICTION_TIME,
]


class EvaluationWriter(OutputWriter, ABC):
    """
    An abstract base class for all classes that evaluate the predictions provided by a learner and allow to write the
    evaluation results to one or several sinks.
    """

    KWARG_FOLD = 'fold'

    class EvaluationResult(Formattable, Tabularizable):
        """
        Stores the evaluation results according to different measures.
        """

        def __init__(self):
            self.measures: Set[Formatter] = set()
            self.results: Optional[List[Dict[Formatter, float]]] = None

        def put(self, measure: Formatter, score: float, num_folds: int, fold: Optional[int]):
            """
            Adds a new score according to a specific measure to the evaluation result.

            :param measure:     The measure
            :param score:       The score according to the measure
            :param num_folds:   The total number of cross validation folds
            :param fold:        The fold, the score corresponds to, or None, if no cross validation is used
            """
            if self.results is None:
                self.results = [{} for _ in range(num_folds)]
            elif len(self.results) != num_folds:
                raise AssertionError('Inconsistent number of total folds given')

            self.measures.add(measure)
            values = self.results[fold if fold is not None else 0]
            values[measure] = score

        def get(self, measure: Formatter, fold: Optional[int], **kwargs) -> str:
            """
            Returns the score according to a specific measure.

            :param measure: The measure
            :param fold:    The fold, the score corresponds to, or None, if no cross validation is used
            :return:        A textual representation of the score
            """
            if self.results is None:
                raise AssertionError('No evaluation results available')

            score = self.results[fold if fold is not None else 0][measure]
            return measure.format(score, **kwargs)

        def dict(self, fold: Optional[int], **kwargs) -> Dict[Formatter, str]:
            """
            Returns a dictionary that stores the scores for a specific fold according to each measure.

            :param fold:    The fold, the scores correspond to, or None, if no cross validation is used
            :return:        A dictionary that stores textual representations of the scores for the given fold according
                            to each measure
            """
            if self.results is None:
                raise AssertionError('No evaluation results available')

            results: Dict[Formatter, str] = {}

            for measure, score in self.results[fold if fold is not None else 0].items():
                results[measure] = measure.format(score, **kwargs)

            return results

        def avg(self, measure: Formatter, **kwargs) -> Tuple[str, str]:
            """
            Returns the score and standard deviation according to a specific measure averaged over all available folds.

            :param measure: The measure
            :return:        A tuple consisting of textual representations of the averaged score and standard deviation
            """
            values = []

            for i in range(len(self.results)):
                results = self.results[i]

                if len(results) > 0:
                    values.append(results[measure])

            values = np.array(values)
            return measure.format(np.average(values), **kwargs), measure.format(np.std(values), **kwargs)

        def avg_dict(self, **kwargs) -> Dict[Formatter, str]:
            """
            Returns a dictionary that stores the scores, averaged across all folds, as well as the standard deviation,
            according to each measure.

            :return: A dictionary that stores textual representations of the scores and standard deviation according to
                     each measure
            """
            result: Dict[Formatter, str] = {}

            for measure in self.measures:
                score, std_dev = self.avg(measure, **kwargs)
                result[measure] = score
                result[Formatter(measure.option, 'Std.-dev. ' + measure.name, measure.percentage)] = std_dev

            return result

        def format(self, options: Options, **kwargs) -> str:
            fold = kwargs.get(EvaluationWriter.KWARG_FOLD)
            percentage = options.get_bool(OPTION_PERCENTAGE, True)
            decimals = options.get_int(OPTION_DECIMALS, 2)
            enable_all = options.get_bool(OPTION_ENABLE_ALL, True)
            rows = []

            for measure in sorted(self.measures):
                if options.get_bool(measure.option, enable_all) and measure != EVALUATION_MEASURE_TRAINING_TIME \
                    and measure != EVALUATION_MEASURE_PREDICTION_TIME:
                    if fold is None:
                        score, std_dev = self.avg(measure, percentage=percentage, decimals=decimals)
                        rows.append([str(measure), score, '±' + std_dev])
                    else:
                        score = self.get(measure, fold, percentage=percentage, decimals=decimals)
                        rows.append([str(measure), score])

            return format_table(rows)

        def tabularize(self, options: Options, **kwargs) -> Optional[List[Dict[str, str]]]:
            fold = kwargs.get(EvaluationWriter.KWARG_FOLD)
            percentage = options.get_bool(OPTION_PERCENTAGE, True)
            decimals = options.get_int(OPTION_DECIMALS, 0)
            enable_all = options.get_bool(OPTION_ENABLE_ALL, True)

            if fold is None:
                columns = self.avg_dict(percentage=percentage, decimals=decimals)
            else:
                columns = self.dict(fold, percentage=percentage, decimals=decimals)

            filtered_columns = {}

            for measure, value in columns.items():
                if options.get_bool(measure.option, enable_all):
                    filtered_columns[measure.name] = value

            return [filtered_columns]

    class LogSink(OutputWriter.LogSink):
        """
        Allows to write evaluation results to the console.
        """

        def __init__(self, options: Options = Options()):
            super().__init__(title='Evaluation result', options=options)

        def write_output(self, meta_data: MetaData, data_split: DataSplit, data_type: Optional[DataType],
                         prediction_scope: Optional[PredictionScope], output_data, **kwargs):
            fold = data_split.get_fold() if data_split.is_cross_validation_used() else 0
            new_kwargs = {**kwargs, **{EvaluationWriter.KWARG_FOLD: fold}}
            super().write_output(meta_data, data_split, data_type, prediction_scope, output_data, **new_kwargs)

            if data_split.is_cross_validation_used() and data_split.is_last_fold():
                super().write_output(meta_data, CrossValidationOverall(data_split.get_num_folds()), data_type,
                                     prediction_scope, output_data, **kwargs)

    class CsvSink(OutputWriter.CsvSink):
        """
        Allows to write evaluation results to CSV files.
        """

        def __init__(self, output_dir: str, options: Options = Options()):
            super().__init__(output_dir=output_dir, file_name='evaluation', options=options)

        def write_output(self, meta_data: MetaData, data_split: DataSplit, data_type: Optional[DataType],
                         prediction_scope: Optional[PredictionScope], output_data, **kwargs):
            fold = data_split.get_fold() if data_split.is_cross_validation_used() else 0
            new_kwargs = {**kwargs, **{EvaluationWriter.KWARG_FOLD: fold}}
            super().write_output(meta_data, data_split, data_type, prediction_scope, output_data, **new_kwargs)

            if data_split.is_cross_validation_used() and data_split.is_last_fold():
                super().write_output(meta_data, CrossValidationOverall(data_split.get_num_folds()), data_type,
                                     prediction_scope, output_data, **kwargs)

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks)
        self.results: Dict[str, EvaluationWriter.EvaluationResult] = {}

    def _generate_output_data(self, meta_data: MetaData, x, y, data_split: DataSplit, learner,
                              data_type: Optional[DataType], prediction_type: Optional[PredictionType],
                              prediction_scope: Optional[PredictionScope], predictions: Optional[Any],
                              train_time: float, predict_time: float) -> Optional[Any]:
        result = self.results[data_type] if data_type in self.results else EvaluationWriter.EvaluationResult()
        self.results[data_type] = result
        num_folds = data_split.get_num_folds()
        fold = data_split.get_fold()
        result.put(EVALUATION_MEASURE_TRAINING_TIME, train_time, num_folds=num_folds, fold=fold)
        result.put(EVALUATION_MEASURE_PREDICTION_TIME, predict_time, num_folds=num_folds, fold=fold)
        self._populate_result(data_split, result, predictions, y)
        return result


class BinaryEvaluationWriter(EvaluationWriter):
    """
    Evaluates the quality of binary predictions provided by a single- or multi-label classifier according to commonly
    used bipartition measures.
    """

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks)
        options = [sink.options for sink in sinks]
        self.multi_Label_evaluation_functions = filter_formatters(MULTI_LABEL_EVALUATION_MEASURES, options)
        self.single_Label_evaluation_functions = filter_formatters(SINGLE_LABEL_EVALUATION_MEASURES, options)

    def _populate_result(self, data_split: DataSplit, result: EvaluationWriter.EvaluationResult, predictions,
                         ground_truth):
        num_folds = data_split.get_num_folds()
        fold = data_split.get_fold()

        if is_multilabel(ground_truth):
            evaluation_functions = self.multi_Label_evaluation_functions
        else:
            predictions = np.ravel(enforce_dense(predictions, order='C', dtype=DTYPE_UINT8))
            ground_truth = np.ravel(enforce_dense(ground_truth, order='C', dtype=DTYPE_UINT8))
            evaluation_functions = self.single_Label_evaluation_functions

        for evaluation_function in evaluation_functions:
            if isinstance(evaluation_function, EvaluationFunction):
                score = evaluation_function.evaluate(ground_truth, predictions)
                result.put(evaluation_function, score, num_folds=num_folds, fold=fold)


class ScoreEvaluationWriter(EvaluationWriter):
    """
    Evaluates the quality of regression scores provided by a single- or multi-output regressor according to commonly
    used regression and ranking measures.
    """

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks)
        options = [sink.options for sink in sinks]
        self.regression_evaluation_functions = filter_formatters(REGRESSION_EVALUATION_MEASURES, options)
        self.ranking_evaluation_functions = filter_formatters(RANKING_EVALUATION_MEASURES, options)

    def _populate_result(self, data_split: DataSplit, result: EvaluationWriter.EvaluationResult, predictions,
                         ground_truth):
        num_folds = data_split.get_num_folds()
        fold = data_split.get_fold()
        ground_truth = enforce_dense(ground_truth, order='C', dtype=DTYPE_UINT8)

        if is_multilabel(ground_truth):
            evaluation_functions = self.ranking_evaluation_functions + self.regression_evaluation_functions
        else:
            evaluation_functions = self.regression_evaluation_functions

            if predictions.shape[1] > 1:
                predictions = predictions[:, -1]

        for evaluation_function in evaluation_functions:
            if isinstance(evaluation_function, EvaluationFunction):
                score = evaluation_function.evaluate(ground_truth, predictions)
                result.put(evaluation_function, score, num_folds=num_folds, fold=fold)


class ProbabilityEvaluationWriter(ScoreEvaluationWriter):
    """
    Evaluates the quality of probability estimates provided by a single- or multi-label classifier according to commonly
    used regression and ranking measures.
    """

    def __init__(self, sinks: List[OutputWriter.Sink]):
        super().__init__(sinks)
