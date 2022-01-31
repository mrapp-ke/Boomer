#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions for parsing command line arguments.
"""
import logging as log
from argparse import ArgumentParser
from enum import Enum

from mlrl.common.options import BooleanOption
from mlrl.common.rule_learners import SparsePolicy, RULE_INDUCTION_VALUES, LABEL_SAMPLING_VALUES, \
    FEATURE_SAMPLING_VALUES, INSTANCE_SAMPLING_VALUES, PARTITION_SAMPLING_VALUES, \
    PRUNING_VALUES, PARALLEL_VALUES
from mlrl.common.strings import format_enum_values, format_string_set, format_dict_keys

PARAM_LOG_LEVEL = '--log-level'

PARAM_CURRENT_FOLD = '--current-fold'

PARAM_RANDOM_STATE = '--random-state'

PARAM_DATA_DIR = '--data-dir'

PARAM_DATASET = '--dataset'

PARAM_FOLDS = '--folds'

PARAM_PRINT_EVALUATION = '--print-evaluation'

PARAM_STORE_EVALUATION = '--store-evaluation'

PARAM_EVALUATE_TRAINING_DATA = '--evaluate-training-data'

PARAM_ONE_HOT_ENCODING = '--one-hot-encoding'

PARAM_MODEL_DIR = '--model-dir'

PARAM_PARAMETER_DIR = '--parameter-dir'

PARAM_OUTPUT_DIR = '--output-dir'

PARAM_STORE_PREDICTIONS = '--store-predictions'

PARAM_PRINT_DATA_CHARACTERISTICS = '--print-data-characteristics'

PARAM_STORE_DATA_CHARACTERISTICS = '--store-data-characteristics'

PARAM_PRINT_MODEL_CHARACTERISTICS = '--print-model-characteristics'

PARAM_STORE_MODEL_CHARACTERISTICS = '--store-model-characteristics'

PARAM_PRINT_RULES = '--print-rules'

PARAM_STORE_RULES = '--store-rules'

PARAM_PRINT_OPTIONS = '--print-options'

PARAM_FEATURE_FORMAT = '--feature-format'

PARAM_LABEL_FORMAT = '--label-format'

PARAM_PREDICTION_FORMAT = '--prediction-format'

PARAM_MAX_RULES = '--max-rules'

PARAM_TIME_LIMIT = '--time-limit'

PARAM_LABEL_SAMPLING = '--label-sampling'

PARAM_FEATURE_SAMPLING = '--feature-sampling'

PARAM_PARTITION_SAMPLING = '--holdout'

PARAM_PRUNING = '--pruning'

PARAM_RULE_INDUCTION = '--rule-induction'

PARAM_PARALLEL_RULE_REFINEMENT = '--parallel-rule-refinement'

PARAM_PARALLEL_STATISTIC_UPDATE = '--parallel-statistic-update'

PARAM_PARALLEL_PREDICTION = '--parallel-prediction'

PARAM_INSTANCE_SAMPLING = '--instance-sampling'

PARAM_HEAD_TYPE = '--head-type'


class LogLevel(Enum):
    DEBUG = 'debug'
    INFO = 'info'
    WARN = 'warn'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'
    FATAL = 'fatal'
    NOTSET = 'notset'


def log_level(s):
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
    raise ValueError(
        'Invalid value given for parameter "' + PARAM_LOG_LEVEL + '". Must be one of ' + format_enum_values(
            LogLevel) + ', but is "' + str(s) + '".')


def current_fold_string(s):
    n = int(s)
    if n >= 0:
        return n - 1
    raise ValueError(
        'Invalid value given for parameter "' + PARAM_CURRENT_FOLD + '". Must be at least 0, but is "' + str(n) + '".')


def boolean_string(s):
    return BooleanOption.parse(s)


def optional_string(s):
    if s is None or s == 'none':
        return None
    return s


def add_log_level_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_LOG_LEVEL, type=log_level, default=LogLevel.INFO.value,
                        help='The log level to be used. Must be one of ' + format_enum_values(LogLevel) + '.')


def add_random_state_argument(parser: ArgumentParser):
    parser.add_argument(PARAM_RANDOM_STATE, type=int, default=1,
                        help='The seed to be used by random number generators. Must be at least 1.')


def add_learner_arguments(parser: ArgumentParser):
    parser.add_argument(PARAM_DATA_DIR, type=str, required=True,
                        help='The path of the directory where the data set files are located.')
    parser.add_argument(PARAM_DATASET, type=str, required=True,
                        help='The name of the data set files without suffix.')
    parser.add_argument(PARAM_FOLDS, type=int, default=1,
                        help='The total number of folds to be used for cross validation. Must be greater than 1 or 1, '
                             + 'if no cross validation should be used.')
    parser.add_argument(PARAM_CURRENT_FOLD, type=current_fold_string, default=-1,
                        help='The cross validation fold to be performed. Must be in [1, ' + PARAM_FOLDS + '] or 0, if '
                             + 'all folds should be performed. This parameter is ignored if ' + PARAM_FOLDS + ' is set '
                             + 'to 1.')
    parser.add_argument(PARAM_PRINT_EVALUATION, type=boolean_string, default=True,
                        help='Whether the evaluation results should be printed on the console or not. Must be one of '
                             + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_STORE_EVALUATION, type=boolean_string, default=True,
                        help='Whether the evaluation results should be written into output files or not. Must be one '
                             + 'of ' + format_enum_values(BooleanOption) + '. Does only have an effect if the '
                             + 'parameter ' + PARAM_OUTPUT_DIR + ' is specified')
    parser.add_argument(PARAM_EVALUATE_TRAINING_DATA, type=boolean_string, default=False,
                        help='Whether the models should not only be evaluated on the test data, but also on the '
                             + 'training data. Must be one of ' + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_PRINT_DATA_CHARACTERISTICS, type=boolean_string, default=False,
                        help='Whether the characteristics of the training data should be printed on the console or '
                             + 'not. Must be one of ' + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_STORE_DATA_CHARACTERISTICS, type=boolean_string, default=False,
                        help='Whether the characteristics of the training data should be written into output files or '
                             + 'not. Must be one of ' + format_enum_values(BooleanOption) + '. Does only have an '
                             + 'effect if the parameter ' + PARAM_OUTPUT_DIR + ' is specified.')
    parser.add_argument(PARAM_ONE_HOT_ENCODING, type=boolean_string, default=False,
                        help='Whether one-hot-encoding should be used to encode nominal attributes or not. Must be one '
                             + 'of ' + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_MODEL_DIR, type=str,
                        help='The path of the directory where models should be stored.')
    parser.add_argument(PARAM_PARAMETER_DIR, type=str,
                        help='The path of the directory where configuration files, which specify the parameters to be '
                             + 'used by the algorithm, are located.')
    parser.add_argument(PARAM_OUTPUT_DIR, type=str,
                        help='The path of the directory where experimental results should be saved.')
    parser.add_argument(PARAM_STORE_PREDICTIONS, type=boolean_string, default=False,
                        help='Whether the predictions for individual examples and labels should be written into output '
                             + 'files or not. Must be one of ' + format_enum_values(BooleanOption) + '. Does only have '
                             + 'an effect, if the parameter ' + PARAM_OUTPUT_DIR + ' is specified.')


def add_rule_learner_arguments(parser: ArgumentParser):
    add_log_level_argument(parser)
    add_random_state_argument(parser)
    add_learner_arguments(parser)
    parser.add_argument(PARAM_PRINT_MODEL_CHARACTERISTICS, type=boolean_string, default=False,
                        help='Whether the characteristics of models should be printed on the console or not. Must be '
                             + 'one of ' + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_STORE_MODEL_CHARACTERISTICS, type=boolean_string, default=False,
                        help='Whether the characteristics of models should be written into output files or not. Must '
                             + 'be one of ' + format_enum_values(BooleanOption) + '. Does only have an effect if the '
                             + 'parameter ' + PARAM_OUTPUT_DIR + ' is specified.')
    parser.add_argument(PARAM_PRINT_RULES, type=boolean_string, default=False,
                        help='Whether the induced rules should be printed on the console or not. Must be one of '
                             + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_STORE_RULES, type=boolean_string, default=False,
                        help='Whether the induced rules should be written into a text file or not. Must be one of '
                             + format_enum_values(BooleanOption) + '. Does only have an effect if the parameter '
                             + PARAM_OUTPUT_DIR + ' is specified.')
    parser.add_argument(PARAM_PRINT_OPTIONS, type=optional_string,
                        help='Additional options to be taken into account when writing rules to the console or an '
                             + 'output file. Does only have an effect if the parameter ' + PARAM_PRINT_RULES + ' or '
                             + PARAM_STORE_RULES + ' is set to True. For a list of the available options refer to the '
                             + 'documentation.')
    parser.add_argument(PARAM_FEATURE_FORMAT, type=str, default=SparsePolicy.AUTO.value,
                        help='The format to be used for the representation of the feature matrix. Must be one of '
                             + format_enum_values(SparsePolicy) + '.')
    parser.add_argument(PARAM_LABEL_FORMAT, type=str, default=SparsePolicy.AUTO.value,
                        help='The format to be used for the representation of the label matrix. Must be one of '
                             + format_enum_values(SparsePolicy) + '.')
    parser.add_argument(PARAM_PREDICTION_FORMAT, type=str, default=SparsePolicy.AUTO.value,
                        help='The format to be used for the representation of predicted labels. Must be one of '
                             + format_enum_values(SparsePolicy) + '.')
    parser.add_argument(PARAM_MAX_RULES, type=int,
                        help='The maximum number of rules to be induced. Must be at least 1 or 0, if the number of '
                             + 'rules should not be restricted.')
    parser.add_argument(PARAM_TIME_LIMIT, type=int,
                        help='The duration in seconds after which the induction of rules should be canceled. Must be '
                             + 'at least 1 or 0, if no time limit should be set.')
    parser.add_argument(PARAM_LABEL_SAMPLING, type=str,
                        help='The name of the strategy to be used for label sampling. Must be one of '
                             + format_dict_keys(LABEL_SAMPLING_VALUES) + '. For additional options refer to the '
                             + 'documentation.')
    parser.add_argument(PARAM_INSTANCE_SAMPLING, type=str,
                        help='The name of the strategy to be used for instance sampling. Must be one of'
                             + format_dict_keys(INSTANCE_SAMPLING_VALUES) + '. For additional options refer to the '
                             + 'documentation.')
    parser.add_argument(PARAM_FEATURE_SAMPLING, type=str,
                        help='The name of the strategy to be used for feature sampling. Must be one of '
                             + format_dict_keys(FEATURE_SAMPLING_VALUES) + '. For additional options refer to the '
                             + 'documentation.')
    parser.add_argument(PARAM_PARTITION_SAMPLING, type=str,
                        help='The name of the strategy to be used for creating a holdout set. Must be one of '
                             + format_dict_keys(PARTITION_SAMPLING_VALUES) + '. For additional options refer to the '
                             + 'documentation.')
    parser.add_argument(PARAM_PRUNING, type=str,
                        help='The name of the strategy to be used for pruning rules. Must be one of '
                             + format_string_set(PRUNING_VALUES) + '. Does only have an effect if the parameter '
                             + PARAM_INSTANCE_SAMPLING + ' is not set to "none".')
    parser.add_argument(PARAM_RULE_INDUCTION, type=str,
                        help='The name of the algorithm to be used for the induction of individual rules. Must be one '
                             + 'of ' + format_string_set(RULE_INDUCTION_VALUES) + '. For additional options refer to '
                             + 'the documentation')
    parser.add_argument(PARAM_PARALLEL_PREDICTION, type=str,
                        help='Whether predictions for different examples should be obtained in parallel or not. Must '
                             + 'be one of ' + format_dict_keys(PARALLEL_VALUES) + '. For additional options refer to '
                             + 'the documentation.')
