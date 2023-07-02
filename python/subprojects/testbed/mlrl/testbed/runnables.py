"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""
import logging as log
import sys

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from enum import Enum
from typing import Dict, List, Optional, Set

from mlrl.common.config import NONE, Parameter, configure_argument_parser, create_kwargs_from_parameters
from mlrl.common.cython.validation import assert_greater, assert_greater_or_equal, assert_less, assert_less_or_equal
from mlrl.common.format import format_dict_keys, format_enum_values
from mlrl.common.options import BooleanOption, parse_param_and_options
from mlrl.common.rule_learners import SparsePolicy

from mlrl.testbed.characteristics import OPTION_DISTINCT_LABEL_VECTORS, OPTION_LABEL_CARDINALITY, \
    OPTION_LABEL_DENSITY, OPTION_LABEL_IMBALANCE_RATIO, OPTION_LABEL_SPARSITY, OPTION_LABELS
from mlrl.testbed.data_characteristics import OPTION_EXAMPLES, OPTION_FEATURE_DENSITY, OPTION_FEATURE_SPARSITY, \
    OPTION_FEATURES, OPTION_NOMINAL_FEATURES, OPTION_NUMERICAL_FEATURES, DataCharacteristicsWriter
from mlrl.testbed.data_splitting import CrossValidationSplitter, DataSet, DataSplitter, NoSplitter, TrainTestSplitter
from mlrl.testbed.evaluation import OPTION_ACCURACY, OPTION_COVERAGE_ERROR, OPTION_DISCOUNTED_CUMULATIVE_GAIN, \
    OPTION_ENABLE_ALL, OPTION_EXAMPLE_WISE_F1, OPTION_EXAMPLE_WISE_JACCARD, OPTION_EXAMPLE_WISE_PRECISION, \
    OPTION_EXAMPLE_WISE_RECALL, OPTION_F1, OPTION_HAMMING_ACCURACY, OPTION_HAMMING_LOSS, OPTION_JACCARD, \
    OPTION_LABEL_RANKING_AVERAGE_PRECISION, OPTION_MACRO_F1, OPTION_MACRO_JACCARD, OPTION_MACRO_PRECISION, \
    OPTION_MACRO_RECALL, OPTION_MEAN_ABSOLUTE_ERROR, OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, OPTION_MEAN_SQUARED_ERROR, \
    OPTION_MEDIAN_ABSOLUTE_ERROR, OPTION_MICRO_F1, OPTION_MICRO_JACCARD, OPTION_MICRO_PRECISION, OPTION_MICRO_RECALL, \
    OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, OPTION_PRECISION, OPTION_PREDICTION_TIME, OPTION_RANK_LOSS, \
    OPTION_RECALL, OPTION_SUBSET_ACCURACY, OPTION_SUBSET_ZERO_ONE_LOSS, OPTION_TRAINING_TIME, OPTION_ZERO_ONE_LOSS, \
    BinaryEvaluationWriter, EvaluationWriter, ProbabilityEvaluationWriter, ScoreEvaluationWriter
from mlrl.testbed.experiments import Evaluation, Experiment, GlobalEvaluation, IncrementalEvaluation
from mlrl.testbed.format import OPTION_DECIMALS, OPTION_PERCENTAGE
from mlrl.testbed.io import clear_directory
from mlrl.testbed.label_vectors import OPTION_SPARSE, LabelVectorSetWriter, LabelVectorWriter
from mlrl.testbed.model_characteristics import ModelCharacteristicsWriter, RuleModelCharacteristicsWriter
from mlrl.testbed.models import OPTION_DECIMALS_BODY, OPTION_DECIMALS_HEAD, OPTION_PRINT_BODIES, \
    OPTION_PRINT_FEATURE_NAMES, OPTION_PRINT_HEADS, OPTION_PRINT_LABEL_NAMES, OPTION_PRINT_NOMINAL_VALUES, \
    ModelWriter, RuleModelWriter
from mlrl.testbed.output_writer import OutputWriter
from mlrl.testbed.parameters import ParameterCsvInput, ParameterInput, ParameterWriter
from mlrl.testbed.persistence import ModelPersistence
from mlrl.testbed.prediction_characteristics import PredictionCharacteristicsWriter
from mlrl.testbed.prediction_scope import PredictionType
from mlrl.testbed.predictions import PredictionWriter
from mlrl.testbed.probability_calibration import JointProbabilityCalibrationModelWriter, \
    MarginalProbabilityCalibrationModelWriter

LOG_FORMAT = '%(levelname)s %(message)s'


class LogLevel(Enum):
    DEBUG = 'debug'
    INFO = 'info'
    WARN = 'warn'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'
    FATAL = 'fatal'
    NOTSET = 'notset'

    def parse(s):
        s = s.lower()
        if s == LogLevel.DEBUG.value:
            return log.DEBUG
        elif s == LogLevel.INFO.value:
            return log.INFO
        elif s == LogLevel.WARN.value or s == LogLevel.WARNING.value:
            return log.WARN
        elif s == LogLevel.ERROR.value:
            return log.ERROR
        elif s == LogLevel.CRITICAL.value or s == LogLevel.FATAL.value:
            return log.CRITICAL
        elif s == LogLevel.NOTSET.value:
            return log.NOTSET
        raise ValueError('Invalid log level given. Must be one of ' + format_enum_values(LogLevel) + ', but is "'
                         + str(s) + '".')


class Runnable(ABC):
    """
    A base class for all programs that can be configured via command line arguments.
    """

    def __init__(self, description: str):
        """
        :param description: A description of the program
        """
        self.parser = ArgumentParser(description=description)

    def run(self):
        parser = self.parser
        self._configure_arguments(parser)
        args = parser.parse_args()

        # Configure the logger...
        log_level = args.log_level
        root = log.getLogger()
        root.setLevel(log_level)
        out_handler = log.StreamHandler(sys.stdout)
        out_handler.setLevel(log_level)
        out_handler.setFormatter(log.Formatter(LOG_FORMAT))
        root.addHandler(out_handler)

        self._run(args)

    def _configure_arguments(self, parser: ArgumentParser):
        """
        May be overridden by subclasses in order to configure the command line arguments of the program.

        :param parser:  An `ArgumentParser` that is used for parsing command line arguments
        """
        parser.add_argument('--log-level',
                            type=LogLevel.parse,
                            default=LogLevel.INFO.value,
                            help='The log level to be used. Must be one of ' + format_enum_values(LogLevel) + '.')

    @abstractmethod
    def _run(self, args):
        """
        Must be implemented by subclasses in order to run the program.

        :param args: The command line arguments
        """
        pass


class LearnerRunnable(Runnable, ABC):
    """
    A base class for all programs that perform an experiment that involves training and evaluation of a learner.
    """

    class ClearOutputDirHook(Experiment.ExecutionHook):
        """
        Deletes all files from the output directory before an experiment starts.
        """

        def __init__(self, output_dir: str):
            self.output_dir = output_dir

        def execute(self):
            clear_directory(self.output_dir)

    PARAM_DATA_SPLIT = '--data-split'

    DATA_SPLIT_TRAIN_TEST = 'train-test'

    OPTION_TEST_SIZE = 'test_size'

    DATA_SPLIT_CROSS_VALIDATION = 'cross-validation'

    OPTION_NUM_FOLDS = 'num_folds'

    OPTION_CURRENT_FOLD = 'current_fold'

    DATA_SPLIT_VALUES: Dict[str, Set[str]] = {
        NONE: {},
        DATA_SPLIT_TRAIN_TEST: {OPTION_TEST_SIZE},
        DATA_SPLIT_CROSS_VALIDATION: {OPTION_NUM_FOLDS, OPTION_CURRENT_FOLD}
    }

    PARAM_PRINT_EVALUATION = '--print-evaluation'

    PRINT_EVALUATION_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            OPTION_ENABLE_ALL, OPTION_HAMMING_LOSS, OPTION_HAMMING_ACCURACY, OPTION_SUBSET_ZERO_ONE_LOSS,
            OPTION_SUBSET_ACCURACY, OPTION_MICRO_PRECISION, OPTION_MICRO_RECALL, OPTION_MICRO_F1, OPTION_MICRO_JACCARD,
            OPTION_MACRO_PRECISION, OPTION_MACRO_RECALL, OPTION_MACRO_F1, OPTION_MACRO_JACCARD,
            OPTION_EXAMPLE_WISE_PRECISION, OPTION_EXAMPLE_WISE_RECALL, OPTION_EXAMPLE_WISE_F1,
            OPTION_EXAMPLE_WISE_JACCARD, OPTION_ACCURACY, OPTION_ZERO_ONE_LOSS, OPTION_PRECISION, OPTION_RECALL,
            OPTION_F1, OPTION_JACCARD, OPTION_MEAN_ABSOLUTE_ERROR, OPTION_MEAN_SQUARED_ERROR,
            OPTION_MEDIAN_ABSOLUTE_ERROR, OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, OPTION_RANK_LOSS,
            OPTION_COVERAGE_ERROR, OPTION_LABEL_RANKING_AVERAGE_PRECISION, OPTION_DISCOUNTED_CUMULATIVE_GAIN,
            OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, OPTION_DECIMALS, OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_EVALUATION = '--store-evaluation'

    STORE_EVALUATION_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            OPTION_ENABLE_ALL, OPTION_HAMMING_LOSS, OPTION_HAMMING_ACCURACY, OPTION_SUBSET_ZERO_ONE_LOSS,
            OPTION_SUBSET_ACCURACY, OPTION_MICRO_PRECISION, OPTION_MICRO_RECALL, OPTION_MICRO_F1, OPTION_MICRO_JACCARD,
            OPTION_MACRO_PRECISION, OPTION_MACRO_RECALL, OPTION_MACRO_F1, OPTION_MACRO_JACCARD,
            OPTION_EXAMPLE_WISE_PRECISION, OPTION_EXAMPLE_WISE_RECALL, OPTION_EXAMPLE_WISE_F1,
            OPTION_EXAMPLE_WISE_JACCARD, OPTION_ACCURACY, OPTION_ZERO_ONE_LOSS, OPTION_PRECISION, OPTION_RECALL,
            OPTION_F1, OPTION_JACCARD, OPTION_MEAN_ABSOLUTE_ERROR, OPTION_MEAN_SQUARED_ERROR,
            OPTION_MEDIAN_ABSOLUTE_ERROR, OPTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR, OPTION_RANK_LOSS,
            OPTION_COVERAGE_ERROR, OPTION_LABEL_RANKING_AVERAGE_PRECISION, OPTION_DISCOUNTED_CUMULATIVE_GAIN,
            OPTION_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN, OPTION_TRAINING_TIME, OPTION_PREDICTION_TIME, OPTION_DECIMALS,
            OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_PRINT_PREDICTIONS = '--print-predictions'

    PRINT_PREDICTIONS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_PREDICTIONS = '--store-predictions'

    STORE_PREDICTIONS_VALUES = PRINT_PREDICTIONS_VALUES

    PARAM_PRINT_PREDICTION_CHARACTERISTICS = '--print-prediction-characteristics'

    PRINT_PREDICTION_CHARACTERISTICS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            OPTION_LABELS, OPTION_LABEL_DENSITY, OPTION_LABEL_SPARSITY, OPTION_LABEL_IMBALANCE_RATIO,
            OPTION_LABEL_CARDINALITY, OPTION_DISTINCT_LABEL_VECTORS, OPTION_DECIMALS, OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_PREDICTION_CHARACTERISTICS = '--store-prediction-characteristics'

    STORE_PREDICTION_CHARACTERISTICS_VALUES = PRINT_PREDICTION_CHARACTERISTICS_VALUES

    PARAM_PRINT_DATA_CHARACTERISTICS = '--print-data-characteristics'

    PRINT_DATA_CHARACTERISTICS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            OPTION_EXAMPLES, OPTION_FEATURES, OPTION_NUMERICAL_FEATURES, OPTION_NOMINAL_FEATURES,
            OPTION_FEATURE_DENSITY, OPTION_FEATURE_SPARSITY, OPTION_LABELS, OPTION_LABEL_DENSITY, OPTION_LABEL_SPARSITY,
            OPTION_LABEL_IMBALANCE_RATIO, OPTION_LABEL_CARDINALITY, OPTION_DISTINCT_LABEL_VECTORS, OPTION_DECIMALS,
            OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_DATA_CHARACTERISTICS = '--store-data-characteristics'

    STORE_DATA_CHARACTERISTICS_VALUES = PRINT_DATA_CHARACTERISTICS_VALUES

    PARAM_PRINT_LABEL_VECTORS = '--print-label-vectors'

    PRINT_LABEL_VECTORS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_SPARSE},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_LABEL_VECTORS = '--store-label-vectors'

    STORE_LABEL_VECTORS_VALUES = PRINT_LABEL_VECTORS_VALUES

    PARAM_OUTPUT_DIR = '--output-dir'

    PARAM_PREDICTION_TYPE = '--prediction-type'

    def __init__(self, description: str, learner_name: str):
        """
        :param learner_name: The name of the learner
        """
        super().__init__(description)
        self.learner_name = learner_name

    def __create_prediction_type(self, args) -> PredictionType:
        prediction_type = args.prediction_type

        try:
            return PredictionType(prediction_type)
        except ValueError:
            raise ValueError('Invalid value given for parameter "' + self.PARAM_PREDICTION_TYPE + '": Must be one of '
                             + format_enum_values(PredictionType) + ', but is "' + str(prediction_type) + '"')

    def __create_data_splitter(self, args) -> DataSplitter:
        data_set = DataSet(data_dir=args.data_dir,
                           data_set_name=args.dataset,
                           use_one_hot_encoding=args.one_hot_encoding)
        value, options = parse_param_and_options(self.PARAM_DATA_SPLIT, args.data_split, self.DATA_SPLIT_VALUES)

        if value == self.DATA_SPLIT_CROSS_VALIDATION:
            num_folds = options.get_int(self.OPTION_NUM_FOLDS, 10)
            assert_greater_or_equal(self.OPTION_NUM_FOLDS, num_folds, 2)
            current_fold = options.get_int(self.OPTION_CURRENT_FOLD, 0)
            if current_fold != 0:
                assert_greater_or_equal(self.OPTION_CURRENT_FOLD, current_fold, 1)
                assert_less_or_equal(self.OPTION_CURRENT_FOLD, current_fold, num_folds)
            return CrossValidationSplitter(data_set,
                                           num_folds=num_folds,
                                           current_fold=current_fold - 1,
                                           random_state=args.random_state)
        elif value == self.DATA_SPLIT_TRAIN_TEST:
            test_size = options.get_float(self.OPTION_TEST_SIZE, 0.33)
            assert_greater(self.OPTION_TEST_SIZE, test_size, 0)
            assert_less(self.OPTION_TEST_SIZE, test_size, 1)
            return TrainTestSplitter(data_set, test_size=test_size, random_state=args.random_state)
        else:
            return NoSplitter(data_set)

    @staticmethod
    def __create_pre_execution_hook(args, data_splitter: DataSplitter) -> Optional[Experiment.ExecutionHook]:
        current_fold = data_splitter.current_fold if isinstance(data_splitter, CrossValidationSplitter) else -1
        return None if args.output_dir is None or current_fold >= 0 else LearnerRunnable.ClearOutputDirHook(
            output_dir=args.output_dir)

    def _configure_arguments(self, parser: ArgumentParser):
        super()._configure_arguments(parser)
        parser.add_argument('--random-state',
                            type=int,
                            default=1,
                            help='The seed to be used by random number generators. Must be at least 1.')
        parser.add_argument('--data-dir',
                            type=str,
                            required=True,
                            help='The path of the directory where the data set files are located.')
        parser.add_argument('--dataset', type=str, required=True, help='The name of the data set files without suffix.')
        parser.add_argument(self.PARAM_DATA_SPLIT,
                            type=str,
                            default=self.DATA_SPLIT_TRAIN_TEST,
                            help='The strategy to be used for splitting the available data into training and test '
                            + 'sets. Must be one of ' + format_dict_keys(self.DATA_SPLIT_VALUES) + '. For additional '
                            + 'options refer to the documentation.')
        parser.add_argument(self.PARAM_PRINT_EVALUATION,
                            type=str,
                            default=BooleanOption.TRUE.value,
                            help='Whether the evaluation results should be printed on the console or not. Must be one '
                            + 'of ' + format_dict_keys(self.PRINT_EVALUATION_VALUES) + '. For additional options refer '
                            + 'to the documentation.')
        parser.add_argument(self.PARAM_STORE_EVALUATION,
                            type=str,
                            default=BooleanOption.TRUE.value,
                            help='Whether the evaluation results should be written into output files or not. Must be '
                            + 'one of ' + format_dict_keys(self.STORE_EVALUATION_VALUES) + '. Does only have an effect '
                            + 'if the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. For additional options '
                            + 'refer to the documentation.')
        parser.add_argument('--evaluate-training-data',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether the models should not only be evaluated on the test data, but also on the '
                            + 'training data. Must be one of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument(self.PARAM_PRINT_PREDICTION_CHARACTERISTICS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of binary predictions should be printed on the console '
                            + 'or not. Must be one of ' + format_dict_keys(self.PRINT_PREDICTION_CHARACTERISTICS_VALUES)
                            + '. Does only have an effect if the parameter ' + self.PARAM_PREDICTION_TYPE + ' is set '
                            + 'to ' + PredictionType.BINARY.value + '. For additional options refer to the '
                            + 'documentation.')
        parser.add_argument(self.PARAM_STORE_PREDICTION_CHARACTERISTICS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of binary predictions should be written into output '
                            + 'files or not. Must be one of '
                            + format_dict_keys(self.STORE_PREDICTION_CHARACTERISTICS_VALUES) + '. Does only have an '
                            + 'effect if the parameter ' + self.PARAM_PREDICTION_TYPE + ' is set to '
                            + PredictionType.BINARY.value + '. For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_PRINT_DATA_CHARACTERISTICS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of the training data should be printed on the console or '
                            + 'not. Must be one of ' + format_dict_keys(self.PRINT_DATA_CHARACTERISTICS_VALUES) + '. '
                            + 'For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_DATA_CHARACTERISTICS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the characteristics of the training data should be written into output files '
                            + 'or not. Must be one of ' + format_dict_keys(self.STORE_DATA_CHARACTERISTICS_VALUES)
                            + '. Does only have an effect if the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. '
                            + 'For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_PRINT_LABEL_VECTORS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the unique label vectors contained in the training data should be printed on '
                            + 'the console or not. Must be one of ' + format_dict_keys(self.PRINT_LABEL_VECTORS_VALUES)
                            + '. For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_LABEL_VECTORS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the unique label vectors contained in the training data should be written '
                            + 'into output files or not. Must be one of '
                            + format_dict_keys(self.STORE_LABEL_VECTORS_VALUES) + '. Does only have an effect if the '
                            + 'parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. For additional options refer to '
                            + 'the documentation.')
        parser.add_argument('--one-hot-encoding',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether one-hot-encoding should be used to encode nominal attributes or not. Must be '
                            + 'one of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument('--model-dir', type=str, help='The path of the directory where models should be stored.')
        parser.add_argument('--parameter-dir',
                            type=str,
                            help='The path of the directory where configuration files, which specify the parameters to '
                            + 'be used by the algorithm, are located.')
        parser.add_argument(self.PARAM_OUTPUT_DIR,
                            type=str,
                            help='The path of the directory where experimental results should be saved.')
        parser.add_argument('--print-parameters',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether the parameter setting should be printed on the console or not. Must be one '
                            + 'of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument('--store-parameters',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether the parameter setting should be written into output files or not. Must be '
                            + 'one of ' + format_enum_values(BooleanOption) + '. Does only have an effect, if the '
                            + 'parameter ' + self.PARAM_OUTPUT_DIR + ' is specified.')
        parser.add_argument(self.PARAM_PRINT_PREDICTIONS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the predictions for individual examples and labels should be printed on the '
                            + 'console or not. Must be one of ' + format_dict_keys(self.PRINT_PREDICTIONS_VALUES) + '. '
                            + 'For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_PREDICTIONS,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the predictions for individual examples and labels should be written into '
                            + 'output files or not. Must be one of ' + format_dict_keys(self.STORE_PREDICTIONS_VALUES)
                            + '. Does only have an effect, if the parameter ' + self.PARAM_OUTPUT_DIR + ' is '
                            + 'specified. For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_PREDICTION_TYPE,
                            type=str,
                            default=PredictionType.BINARY.value,
                            help='The type of predictions that should be obtained from the learner. Must be one of '
                            + format_enum_values(PredictionType) + '.')

    def _run(self, args):
        prediction_type = self.__create_prediction_type(args)
        train_evaluation = self._create_evaluation(
            args, prediction_type,
            self._create_evaluation_output_writers(args, prediction_type) if args.evaluate_training_data else [])
        test_evaluation = self._create_evaluation(args, prediction_type,
                                                  self._create_evaluation_output_writers(args, prediction_type))
        data_splitter = self.__create_data_splitter(args)
        experiment = Experiment(base_learner=self._create_learner(args),
                                learner_name=self.learner_name,
                                data_splitter=data_splitter,
                                pre_training_output_writers=self._create_pre_training_output_writers(args),
                                post_training_output_writers=self._create_post_training_output_writers(args),
                                pre_execution_hook=self.__create_pre_execution_hook(args, data_splitter),
                                train_evaluation=train_evaluation,
                                test_evaluation=test_evaluation,
                                parameter_input=self._create_parameter_input(args),
                                persistence=self._create_persistence(args))
        experiment.run()

    def _create_pre_training_output_writers(self, args) -> List[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter`s that should be invoked before training a
        model.

        :param args:    The command line arguments
        :return:        A list that contains the `OutputWriter`s that have been created
        """
        output_writers = []
        output_writer = self._create_data_characteristics_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        output_writer = self._create_parameter_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        return output_writers

    def _create_post_training_output_writers(self, args) -> List[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter`s that should be invoked after training a
        model.

        :param args:    The command line arguments
        :return:        A list that contains the `OutputWriters`s that have been created
        """
        output_writers = []
        output_writer = self._create_label_vector_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        return output_writers

    def _create_evaluation_output_writers(self, args, prediction_type: PredictionType) -> List[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter`s that should be invoked after evaluating a
        model.

        :param args:            The command line arguments
        :param prediction_type: The type of the predictions
        :return:                A list that contains the `OutputWriter`s that have been created
        """
        output_writers = []
        output_writer = self._create_evaluation_writer(args, prediction_type)

        if output_writer is not None:
            output_writers.append(output_writer)

        output_writer = self._create_prediction_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        output_writer = self._create_prediction_characteristics_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        return output_writers

    def _create_persistence(self, args) -> Optional[ModelPersistence]:
        """
        May be overridden by subclasses in order to create the `ModelPersistence` that should be used to save and load
        models.

        :param args:    The command line arguments
        :return:        The `ModelPersistence` that has been created
        """
        return None if args.model_dir is None else ModelPersistence(model_dir=args.model_dir)

    def _create_evaluation(self, args, prediction_type: PredictionType,
                           output_writers: List[OutputWriter]) -> Optional[Evaluation]:
        """
        May be overridden by subclasses in order to create the `Evaluation` that should be used to evaluate predictions
        that are obtained from a previously trained model.

        :param args:            The command line arguments
        :param prediction_type: The type of the predictions to be obtained
        :param output_writers:  A list that contains all output writers to be invoked after predictions have been
                                obtained
        :return:                The `Evaluation` that has been created
        """
        return GlobalEvaluation(prediction_type, output_writers) if len(output_writers) > 0 else None

    def _create_evaluation_writer(self, args, prediction_type: PredictionType) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output evaluation
        results.

        :param args:            The command line arguments
        :param prediction_type: The type of the predictions
        :return:                The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_EVALUATION, args.print_evaluation,
                                                 self.PRINT_EVALUATION_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(EvaluationWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_EVALUATION, args.store_evaluation,
                                                 self.STORE_EVALUATION_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(EvaluationWriter.CsvSink(output_dir=args.output_dir, options=options))

        if len(sinks) == 0:
            return None
        elif prediction_type == PredictionType.SCORES:
            return ScoreEvaluationWriter(sinks)
        elif prediction_type == PredictionType.PROBABILITIES:
            return ProbabilityEvaluationWriter(sinks)
        else:
            return BinaryEvaluationWriter(sinks)

    def _create_parameter_input(self, args) -> Optional[ParameterInput]:
        """
        May be overridden by subclasses in order to create the `ParameterInput` that should be used to load parameter
        settings.

        :param args:    The command line arguments
        :return:        The `ParameterInput` that has been created
        """
        return None if args.parameter_dir is None else ParameterCsvInput(input_dir=args.parameter_dir)

    def _create_parameter_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output parameter
        settings.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []

        if args.print_parameters:
            sinks.append(ParameterWriter.LogSink())

        if args.store_parameters and args.output_dir is not None:
            sinks.append(ParameterWriter.CsvSink(output_dir=args.output_dir))

        return ParameterWriter(sinks) if len(sinks) > 0 else None

    def _create_prediction_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output predictions.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_PREDICTIONS, args.print_predictions,
                                                 self.PRINT_PREDICTIONS_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(PredictionWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_PREDICTIONS, args.store_predictions,
                                                 self.STORE_PREDICTIONS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(PredictionWriter.ArffSink(output_dir=args.output_dir, options=options))

        return PredictionWriter(sinks) if len(sinks) > 0 else None

    def _create_prediction_characteristics_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output prediction
        characteristics.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_PREDICTION_CHARACTERISTICS,
                                                 args.print_prediction_characteristics,
                                                 self.PRINT_PREDICTION_CHARACTERISTICS_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(PredictionCharacteristicsWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_PREDICTION_CHARACTERISTICS,
                                                 args.store_prediction_characteristics,
                                                 self.STORE_PREDICTION_CHARACTERISTICS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(PredictionCharacteristicsWriter.CsvSink(output_dir=args.output_dir, options=options))

        return PredictionCharacteristicsWriter(sinks) if len(sinks) > 0 else None

    def _create_data_characteristics_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output data
        characteristics.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_DATA_CHARACTERISTICS, args.print_data_characteristics,
                                                 self.PRINT_DATA_CHARACTERISTICS_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(DataCharacteristicsWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_DATA_CHARACTERISTICS, args.store_data_characteristics,
                                                 self.STORE_DATA_CHARACTERISTICS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(DataCharacteristicsWriter.CsvSink(output_dir=args.output_dir, options=options))

        return DataCharacteristicsWriter(sinks) if len(sinks) > 0 else None

    def _create_label_vector_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output unique label
        vectors contained in the training data.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_LABEL_VECTORS, args.print_label_vectors,
                                                 self.PRINT_LABEL_VECTORS_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(LabelVectorWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_LABEL_VECTORS, args.store_label_vectors,
                                                 self.STORE_LABEL_VECTORS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(LabelVectorWriter.CsvSink(output_dir=args.output_dir, options=options))

        return LabelVectorWriter(sinks) if len(sinks) > 0 else None

    @abstractmethod
    def _create_learner(self, args):
        """
        Must be implemented by subclasses in order to create the learner.

        :param args:    The command line arguments
        :return:        The learner that has been created
        """
        pass


class RuleLearnerRunnable(LearnerRunnable):
    """
    A base class for all programs that perform an experiment that involves training and evaluation of a rule learner.
    """

    PARAM_INCREMENTAL_EVALUATION = '--incremental-evaluation'

    OPTION_MIN_SIZE = 'min_size'

    OPTION_MAX_SIZE = 'max_size'

    OPTION_STEP_SIZE = 'step_size'

    INCREMENTAL_EVALUATION_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_MIN_SIZE, OPTION_MAX_SIZE, OPTION_STEP_SIZE},
        BooleanOption.FALSE.value: {}
    }

    PARAM_PRINT_RULES = '--print-rules'

    PRINT_RULES_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            OPTION_PRINT_FEATURE_NAMES, OPTION_PRINT_LABEL_NAMES, OPTION_PRINT_NOMINAL_VALUES, OPTION_PRINT_BODIES,
            OPTION_PRINT_HEADS, OPTION_DECIMALS_BODY, OPTION_DECIMALS_HEAD
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_RULES = '--store-rules'

    STORE_RULES_VALUES = PRINT_RULES_VALUES

    PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL = '--print-marginal-probability-calibration-model'

    PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL = '--store-marginal-probability-calibration-model'

    STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES = PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES

    PARAM_PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL = '--print-joint-probability-calibration-model'

    PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL = '--store-joint-probability-calibration-model'

    STORE_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES = PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES

    def __init__(self, description: str, learner_name: str, learner_type: type, config_type: type,
                 parameters: Set[Parameter]):
        """
        :param learner_type:    The type of the rule learner
        :param config_type:     The type of the rule learner's configuration
        :param parameters:      A set that contains the parameters that may be supported by the rule learner
        """
        super().__init__(description=description, learner_name=learner_name)
        self.learner_type = learner_type
        self.config_type = config_type
        self.parameters = parameters

    def _configure_arguments(self, parser: ArgumentParser):
        super()._configure_arguments(parser)
        parser.add_argument(self.PARAM_INCREMENTAL_EVALUATION,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether models should be evaluated repeatedly, using only a subset of the induced '
                            + 'rules with increasing size, or not. Must be one of ' + format_enum_values(BooleanOption)
                            + '. For additional options refer to the documentation.')
        parser.add_argument('--print-model-characteristics',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether the characteristics of models should be printed on the console or not. Must '
                            + 'be one of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument('--store-model-characteristics',
                            type=BooleanOption.parse,
                            default=False,
                            help='Whether the characteristics of models should be written into output files or not. '
                            + 'Must be one of ' + format_enum_values(BooleanOption) + '. Does only have an effect if '
                            + 'the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified.')
        parser.add_argument(self.PARAM_PRINT_RULES,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the induced rules should be printed on the console or not. Must be one of '
                            + format_dict_keys(self.PRINT_RULES_VALUES) + '. For additional options refer to the '
                            + 'documentation.')
        parser.add_argument(self.PARAM_STORE_RULES,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the induced rules should be written into a text file or not. Must be one of '
                            + format_dict_keys(self.STORE_RULES_VALUES) + '. Does only have an effect if the parameter '
                            + self.PARAM_OUTPUT_DIR + ' is specified. For additional options refer to the '
                            + 'documentation.')
        parser.add_argument(self.PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the model for the calibration of marginal probabilities should be printed on '
                            + 'the console or not. Must be one of ' + format_enum_values(BooleanOption) + '. For '
                            + 'additional options refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the model for the calibration of marginal probabilities should be written '
                            + 'into an output file or not. Must be one of ' + format_enum_values(BooleanOption) + '. '
                            + 'Does only have an effect if the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. '
                            + 'For additional options refer to the documentation.')
        parser.add_argument(self.PARAM_PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the model for the calibration of joint probabilities should be printed on '
                            + 'the console or not. Must be one of ' + format_enum_values(BooleanOption) + '. For '
                            + 'additional options refer to the documentation.')
        parser.add_argument(self.PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL,
                            type=str,
                            default=BooleanOption.FALSE.value,
                            help='Whether the model for the calibration of joint probabilities should be written into '
                            + 'an output file or not. Must be one of ' + format_enum_values(BooleanOption) + '. Does '
                            + 'only have an effect if the parameter ' + self.PARAM_OUTPUT_DIR + ' is specified. For '
                            + 'additional options refer to the documentation.')
        parser.add_argument('--feature-format',
                            type=str,
                            default=SparsePolicy.AUTO.value,
                            help='The format to be used for the representation of the feature matrix. Must be one of '
                            + format_enum_values(SparsePolicy) + '.')
        parser.add_argument('--label-format',
                            type=str,
                            default=SparsePolicy.AUTO.value,
                            help='The format to be used for the representation of the label matrix. Must be one of '
                            + format_enum_values(SparsePolicy) + '.')
        parser.add_argument('--prediction-format',
                            type=str,
                            default=SparsePolicy.AUTO.value,
                            help='The format to be used for the representation of predictions. Must be one of '
                            + format_enum_values(SparsePolicy) + '.')
        configure_argument_parser(parser, self.config_type, self.parameters)

    def _create_learner(self, args):
        kwargs = create_kwargs_from_parameters(args, self.parameters)
        kwargs['random_state'] = args.random_state
        kwargs['feature_format'] = args.feature_format
        kwargs['label_format'] = args.label_format
        kwargs['prediction_format'] = args.prediction_format
        return self.learner_type(**kwargs)

    def _create_model_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output textual
        representations of models.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_RULES, args.print_rules, self.PRINT_RULES_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(ModelWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_RULES, args.store_rules, self.STORE_RULES_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(ModelWriter.TxtSink(output_dir=args.output_dir, options=options))

        return RuleModelWriter(sinks) if len(sinks) > 0 else None

    def _create_model_characteristics_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output the
        characteristics of models.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []

        if args.print_model_characteristics:
            sinks.append(ModelCharacteristicsWriter.LogSink())

        if args.store_model_characteristics and args.output_dir is not None:
            sinks.append(ModelCharacteristicsWriter.CsvSink(output_dir=args.output_dir))

        return RuleModelCharacteristicsWriter(sinks) if len(sinks) > 0 else None

    def _create_marginal_probability_calibration_model_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output textual
        representations of models for the calibration of marginal probabilities.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                                                 args.print_marginal_probability_calibration_model,
                                                 self.PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(MarginalProbabilityCalibrationModelWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                                                 args.store_marginal_probability_calibration_model,
                                                 self.STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(MarginalProbabilityCalibrationModelWriter.CsvSink(output_dir=args.output_dir, options=options))

        return MarginalProbabilityCalibrationModelWriter(sinks) if len(sinks) > 0 else None

    def _create_joint_probability_calibration_model_writer(self, args) -> Optional[OutputWriter]:
        """
        May be overridden by subclasses in order to create the `OutputWriter` that should be used to output textual
        representations of models for the calibration of joint probabilities.

        :param args:    The command line arguments
        :return:        The `OutputWriter` that has been created
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL,
                                                 args.print_joint_probability_calibration_model,
                                                 self.PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(JointProbabilityCalibrationModelWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL,
                                                 args.store_joint_probability_calibration_model,
                                                 self.STORE_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(JointProbabilityCalibrationModelWriter.CsvSink(output_dir=args.output_dir, options=options))

        return JointProbabilityCalibrationModelWriter(sinks) if len(sinks) > 0 else None

    def _create_evaluation(self, args, prediction_type: PredictionType,
                           output_writers: List[OutputWriter]) -> Optional[Evaluation]:
        value, options = parse_param_and_options(self.PARAM_INCREMENTAL_EVALUATION, args.incremental_evaluation,
                                                 self.INCREMENTAL_EVALUATION_VALUES)

        if value == BooleanOption.TRUE.value:
            min_size = options.get_int(self.OPTION_MIN_SIZE, 0)
            assert_greater_or_equal(self.OPTION_MIN_SIZE, min_size, 0)
            max_size = options.get_int(self.OPTION_MAX_SIZE, 0)
            if max_size != 0:
                assert_greater(self.OPTION_MAX_SIZE, max_size, min_size)
            step_size = options.get_int(self.OPTION_STEP_SIZE, 1)
            assert_greater_or_equal(self.OPTION_STEP_SIZE, step_size, 1)
            return IncrementalEvaluation(
                prediction_type, output_writers, min_size=min_size, max_size=max_size,
                step_size=step_size) if len(output_writers) > 0 else None
        else:
            return super()._create_evaluation(args, prediction_type, output_writers)

    def _create_label_vector_writer(self, args) -> Optional[OutputWriter]:
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_LABEL_VECTORS, args.print_label_vectors,
                                                 self.PRINT_LABEL_VECTORS_VALUES)

        if value == BooleanOption.TRUE.value:
            sinks.append(LabelVectorSetWriter.LogSink(options=options))

        value, options = parse_param_and_options(self.PARAM_STORE_LABEL_VECTORS, args.store_label_vectors,
                                                 self.STORE_LABEL_VECTORS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir is not None:
            sinks.append(LabelVectorSetWriter.CsvSink(output_dir=args.output_dir, options=options))

        return LabelVectorSetWriter(sinks) if len(sinks) > 0 else None

    def _create_post_training_output_writers(self, args) -> List[OutputWriter]:
        output_writers = super()._create_post_training_output_writers(args)
        output_writer = self._create_model_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        output_writer = self._create_model_characteristics_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        output_writer = self._create_marginal_probability_calibration_model_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        output_writer = self._create_joint_probability_calibration_model_writer(args)

        if output_writer is not None:
            output_writers.append(output_writer)

        return output_writers

    def _create_model_characteristics_writer(self, args) -> Optional[OutputWriter]:
        sinks = []

        if args.print_model_characteristics:
            sinks.append(ModelCharacteristicsWriter.LogSink())

        if args.store_model_characteristics and args.output_dir is not None:
            sinks.append(ModelCharacteristicsWriter.CsvSink(output_dir=args.output_dir))

        return RuleModelCharacteristicsWriter(sinks) if len(sinks) > 0 else None
