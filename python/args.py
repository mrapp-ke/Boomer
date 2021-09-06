#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides functions for parsing command line arguments.
"""
import logging as log
from argparse import ArgumentParser
from enum import Enum

from mlrl.boosting.boosting_learners import LOSS_LOGISTIC_LABEL_WISE, HEAD_TYPE_VALUES as BOOSTING_HEAD_TYPE_VALUES, \
    EARLY_STOPPING_VALUES, LABEL_BINNING_VALUES, LOSS_VALUES, PREDICTOR_VALUES, \
    PARALLEL_VALUES as BOOSTING_PARALLEL_VALUES
from mlrl.common.options import BooleanOption
from mlrl.common.rule_learners import SparsePolicy, HEAD_TYPE_SINGLE, AUTOMATIC, SAMPLING_WITHOUT_REPLACEMENT, \
    PRUNING_IREP, LABEL_SAMPLING_VALUES, FEATURE_SAMPLING_VALUES, PARTITION_SAMPLING_VALUES, FEATURE_BINNING_VALUES, \
    PRUNING_VALUES, INSTANCE_SAMPLING_VALUES as BOOSTING_INSTANCE_SAMPLING_VALUES, PARALLEL_VALUES
from mlrl.common.strings import format_enum_values, format_string_set, format_dict_keys

PARAM_LOG_LEVEL = '--log-level'

PARAM_CURRENT_FOLD = '--current-fold'

PARAM_RANDOM_STATE = '--random-state'

PARAM_DATA_DIR = '--data-dir'

PARAM_DATASET = '--dataset'

PARAM_FOLDS = '--folds'

PARAM_EVALUATE_TRAINING_DATA = '--evaluate-training-data'

PARAM_ONE_HOT_ENCODING = '--one-hot-encoding'

PARAM_MODEL_DIR = '--model-dir'

PARAM_PARAMETER_DIR = '--parameter-dir'

PARAM_OUTPUT_DIR = '--output-dir'

PARAM_STORE_PREDICTIONS = '--store-predictions'

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

PARAM_FEATURE_BINNING = '--feature-binning'

PARAM_MIN_COVERAGE = '--min-coverage'

PARAM_MAX_CONDITIONS = '--max-conditions'

PARAM_MAX_HEAD_REFINEMENTS = '--max-head-refinements'

PARAM_PARALLEL_RULE_REFINEMENT = '--parallel-rule-refinement'

PARAM_PARALLEL_STATISTIC_UPDATE = '--parallel-statistic-update'

PARAM_PARALLEL_PREDICTION = '--parallel-prediction'

PARAM_DEFAULT_RULE = '--default-rule'

PARAM_RECALCULATE_PREDICTIONS = '--recalculate-predictions'

PARAM_EARLY_STOPPING = '--early-stopping'

PARAM_INSTANCE_SAMPLING = '--instance-sampling'

PARAM_LABEL_BINNING = '--label-binning'

PARAM_SHRINKAGE = '--shrinkage'

PARAM_LOSS = '--loss'

PARAM_PREDICTOR = '--predictor'

PARAM_L2_REGULARIZATION_WEIGHT = '--l2-regularization-weight'

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
    if s is None or s.lower() == 'none':
        return None
    return s


def string_list(s):
    return [x.strip() for x in s.split(',')]


def int_list(s):
    return [int(x) for x in string_list(s)]


def float_list(s):
    return [float(x) for x in string_list(s)]


class ArgumentParserBuilder:
    """
    A builder that allows to configure an `ArgumentParser` that accepts commonly used command-line arguments.
    """

    def __init__(self, description: str, **kwargs):
        parser = ArgumentParser(description=description)
        parser.add_argument(PARAM_LOG_LEVEL, type=log_level,
                            default=ArgumentParserBuilder.__get_or_default('log_level', 'info', **kwargs),
                            help='The log level to be used. Must be one of ' + format_enum_values(LogLevel) + '.')
        self.parser = parser

    def add_random_state_argument(self, **kwargs) -> 'ArgumentParserBuilder':
        self.parser.add_argument(PARAM_RANDOM_STATE, type=int,
                                 default=ArgumentParserBuilder.__get_or_default('random_state', 1, **kwargs),
                                 help='The seed to be used by random number generators. Must be at least 1.')
        return self

    def add_learner_arguments(self, **kwargs) -> 'ArgumentParserBuilder':
        parser = self.parser
        parser.add_argument(PARAM_DATA_DIR, type=str, required=True,
                            help='The path of the directory where the data set files are located.')
        parser.add_argument(PARAM_DATASET, type=str, required=True,
                            help='The name of the data set files without suffix.')
        parser.add_argument(PARAM_FOLDS, type=int, default=1,
                            help='The total number of folds to be used for cross validation. Must be greater than 1 '
                                 + 'or 1, if no cross validation should be used.')
        parser.add_argument(PARAM_CURRENT_FOLD, type=current_fold_string,
                            default=ArgumentParserBuilder.__get_or_default('current_fold', -1, **kwargs),
                            help='The cross validation fold to be performed. Must be in [1, ' + PARAM_FOLDS + '] or 0, '
                                 + 'if all folds should be performed. This parameter is ignored if ' + PARAM_FOLDS
                                 + ' is set to 1.')
        parser.add_argument(PARAM_EVALUATE_TRAINING_DATA, type=boolean_string,
                            default=ArgumentParserBuilder.__get_or_default('evaluate_training_data', False, **kwargs),
                            help='Whether the models should not only be evaluated on the test data, but also on the '
                                 + 'training data. Must be one of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument(PARAM_ONE_HOT_ENCODING, type=boolean_string,
                            default=ArgumentParserBuilder.__get_or_default('one_hot_encoding', False, **kwargs),
                            help='Whether one-hot-encoding should be used to encode nominal attributes or not. Must be '
                                 + 'one of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument(PARAM_MODEL_DIR, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('model_dir', None, **kwargs),
                            help='The path of the directory where models should be stored.')
        parser.add_argument(PARAM_PARAMETER_DIR, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('parameter_dir', None, **kwargs),
                            help='The path of the directory where configuration files, which specify the parameters to '
                                 + 'be used by the algorithm, are located.')
        parser.add_argument(PARAM_OUTPUT_DIR, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('output_dir', None, **kwargs),
                            help='The path of the directory where experimental results should be saved.')
        parser.add_argument(PARAM_STORE_PREDICTIONS, type=boolean_string,
                            default=ArgumentParserBuilder.__get_or_default('store_predictions', False, **kwargs),
                            help='Whether the predictions for individual examples and labels should be written into '
                                 + 'output files or not. Must be one of ' + format_enum_values(BooleanOption) + '. '
                                 + 'Does only have an effect, if the parameter ' + PARAM_OUTPUT_DIR + ' is specified.')
        return self

    def add_rule_learner_arguments(self, **kwargs) -> 'ArgumentParserBuilder':
        self.add_learner_arguments(**kwargs)
        self.add_random_state_argument(**kwargs)
        parser = self.parser
        parser.add_argument(PARAM_PRINT_RULES, type=boolean_string,
                            default=ArgumentParserBuilder.__get_or_default('print_rules', False, **kwargs),
                            help='Whether the induced rules should be printed on the console or not')
        parser.add_argument(PARAM_STORE_RULES, type=boolean_string,
                            default=ArgumentParserBuilder.__get_or_default('store_rules', False, **kwargs),
                            help='Whether the induced rules should be written into a text file or not. Must be one of '
                                 + format_enum_values(BooleanOption) + '. Does only have an effect if the parameter '
                                 + PARAM_OUTPUT_DIR + ' is specified.')
        parser.add_argument(PARAM_PRINT_OPTIONS, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('print_options', None, **kwargs),
                            help='Additional options to be taken into account when writing rules to the console or an '
                                 + 'output file. Does only have an effect if the parameter ' + PARAM_PRINT_RULES
                                 + ' or ' + PARAM_STORE_RULES + ' is set to True. For a list of the available options '
                                 + 'refer to the documentation.')
        parser.add_argument(PARAM_FEATURE_FORMAT, type=optional_string, default=SparsePolicy.AUTO.value,
                            help='The format to be used for the representation of the feature matrix. Must be one of '
                                 + format_enum_values(SparsePolicy) + '.')
        parser.add_argument(PARAM_LABEL_FORMAT, type=optional_string, default=SparsePolicy.AUTO.value,
                            help='The format to be used for the representation of the label matrix. Must be one of '
                                 + format_enum_values(SparsePolicy) + '.')
        parser.add_argument(PARAM_PREDICTION_FORMAT, type=optional_string, default=SparsePolicy.AUTO.value,
                            help='The format to be used for the representation of predicted labels. Must be one of '
                                 + format_enum_values(SparsePolicy) + '.')
        parser.add_argument(PARAM_MAX_RULES, type=int,
                            default=ArgumentParserBuilder.__get_or_default('max_rules', 500, **kwargs),
                            help='The maximum number of rules to be induced. Must be at least 1 or 0, if the number of '
                                 + 'rules should not be restricted.')
        parser.add_argument(PARAM_TIME_LIMIT, type=int,
                            default=ArgumentParserBuilder.__get_or_default('time_limit', 0, **kwargs),
                            help='The duration in seconds after which the induction of rules should be canceled. Must '
                                 + 'be at least 1 or 0, if no time limit should be set.')
        parser.add_argument(PARAM_LABEL_SAMPLING, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('label_sampling', None, **kwargs),
                            help='The name of the strategy to be used for label sampling. Must be one of '
                                 + format_dict_keys(LABEL_SAMPLING_VALUES) + ' or "None", if no label sampling '
                                 + 'should be used. For additional options refer to the documentation.')
        parser.add_argument(PARAM_FEATURE_SAMPLING, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('feature_sampling', None, **kwargs),
                            help='The name of the strategy to be used for feature sampling. Must be one of '
                                 + format_dict_keys(FEATURE_SAMPLING_VALUES) + ' or "None", if no feature sampling '
                                 + 'should be used. For additional options refer to the documentation.')
        parser.add_argument(PARAM_PARTITION_SAMPLING, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('holdout', None, **kwargs),
                            help='The name of the strategy to be used for creating a holdout set. Must be one of '
                                 + format_dict_keys(PARTITION_SAMPLING_VALUES) + ' or "None", if no holdout set '
                                 + 'should be created. For additional options refer to the documentation.')
        parser.add_argument(PARAM_FEATURE_BINNING, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('feature_binning', None, **kwargs),
                            help='The name of the strategy to be used for feature binning. Must be one of '
                                 + format_dict_keys(FEATURE_BINNING_VALUES) + ' or "None", if no feature binning '
                                 + 'should be used. For additional options refer to the documentation.')
        parser.add_argument(PARAM_PRUNING, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('pruning', None, **kwargs),
                            help='The name of the strategy to be used for pruning rules. Must be one of '
                                 + format_string_set(PRUNING_VALUES) + ' or "None", if no pruning should be used. '
                                 + 'Does only have an effect if the parameter ' + PARAM_INSTANCE_SAMPLING + ' is not '
                                 + 'set to "None".')
        parser.add_argument(PARAM_MIN_COVERAGE, type=int,
                            default=ArgumentParserBuilder.__get_or_default('min_coverage', 1, **kwargs),
                            help='The minimum number of training examples that must be covered by a rule. Must be at '
                                 + 'least 1.')
        parser.add_argument(PARAM_MAX_CONDITIONS, type=int,
                            default=ArgumentParserBuilder.__get_or_default('max_conditions', 0, **kwargs),
                            help='The maximum number of conditions to be included in a rule\'s body. Must be at least '
                                 + '1 or 0, if the number of conditions should not be restricted.')
        parser.add_argument(PARAM_MAX_HEAD_REFINEMENTS, type=int,
                            default=ArgumentParserBuilder.__get_or_default('max_head_refinements', 1, **kwargs),
                            help='The maximum number of times the head of a rule may be refined. Must be at least 1 or '
                                 + '0, if the number of refinements should not be restricted.')
        parser.add_argument(PARAM_PARALLEL_PREDICTION, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('parallel_prediction',
                                                                           BooleanOption.TRUE.value, **kwargs),
                            help='Whether predictions for different examples should be obtained in parallel or not. '
                                 + 'Must be one of ' + format_dict_keys(PARALLEL_VALUES) + '. For additional options '
                                 + 'refer to the documentation.')
        return self

    def add_boosting_learner_arguments(self, **kwargs) -> 'ArgumentParserBuilder':
        self.add_rule_learner_arguments(max_rules=1000, feature_sampling=SAMPLING_WITHOUT_REPLACEMENT, **kwargs)
        parser = self.parser
        parser.add_argument(PARAM_DEFAULT_RULE, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('default_rule', BooleanOption.TRUE.value,
                                                                           **kwargs),
                            help='Whether the first rule should be a default rule or not. Must be one of '
                                 + format_enum_values(BooleanOption))
        parser.add_argument(PARAM_RECALCULATE_PREDICTIONS, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('recalculate_predictions',
                                                                           BooleanOption.TRUE.value, **kwargs),
                            help='Whether the predictions of rules should be recalculated on the entire training data, '
                                 + 'if the parameter ' + PARAM_INSTANCE_SAMPLING + ' is not set to None, or not. Must '
                                 + 'be one of ' + format_enum_values(BooleanOption) + '.')
        parser.add_argument(PARAM_EARLY_STOPPING, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('early_stopping', None, **kwargs),
                            help='The name of the strategy to be used for early stopping. Must be one of '
                                 + format_dict_keys(EARLY_STOPPING_VALUES) + ' or "None", if no early stopping '
                                 + 'should be used. Does only have an effect if the parameter '
                                 + PARAM_PARTITION_SAMPLING + ' is not set to "None". For additional options refer to '
                                 + 'the documentation.')
        parser.add_argument(PARAM_INSTANCE_SAMPLING, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('instance_sampling', None, **kwargs),
                            help='The name of the strategy to be used for instance sampling. Must be one of'
                                 + format_dict_keys(BOOSTING_INSTANCE_SAMPLING_VALUES) + ' or "None", if no instance '
                                 + 'sampling should be used. For additional options refer to the documentation.')
        parser.add_argument(PARAM_LABEL_BINNING, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('label_binning', AUTOMATIC, **kwargs),
                            help='The name of the strategy to be used for gradient-based label binning (GBLB). Must be '
                                 + 'one of ' + format_dict_keys(LABEL_BINNING_VALUES) + ' or "None", if no label '
                                 + 'binning should be used. If set to "' + AUTOMATIC + '", the most suitable strategy '
                                 + 'is chosen automatically based on the parameters ' + PARAM_LOSS + ' and '
                                 + PARAM_HEAD_TYPE + '. For additional options refer to the documentation.')
        parser.add_argument(PARAM_SHRINKAGE, type=float,
                            default=ArgumentParserBuilder.__get_or_default('shrinkage', 0.3, **kwargs),
                            help='The shrinkage parameter, a.k.a. the learning rate, to be used. Must be in (0, 1].')
        parser.add_argument(PARAM_LOSS, type=str, default=LOSS_LOGISTIC_LABEL_WISE,
                            help='The name of the loss function to be minimized during training. Must be one of '
                                 + format_string_set(LOSS_VALUES) + '.')
        parser.add_argument(PARAM_PREDICTOR, type=str,
                            default=ArgumentParserBuilder.__get_or_default('predictor', AUTOMATIC, **kwargs),
                            help='The name of the strategy to be used for making predictions. Must be one of '
                                 + format_string_set(PREDICTOR_VALUES) + '. If set to "' + AUTOMATIC + '", the most '
                                 + 'suitable strategy is chosen automatically based on the parameter ' + PARAM_LOSS
                                 + '.')
        parser.add_argument(PARAM_L2_REGULARIZATION_WEIGHT, type=float,
                            default=ArgumentParserBuilder.__get_or_default('l2_regularization_weight', 1.0, **kwargs),
                            help='The weight of the L2 regularization. Must be at least 0.')
        parser.add_argument(PARAM_HEAD_TYPE, type=str,
                            default=ArgumentParserBuilder.__get_or_default('head_type', AUTOMATIC, **kwargs),
                            help='The type of the rule heads that should be used. Must be one of '
                                 + format_string_set(BOOSTING_HEAD_TYPE_VALUES) + '. If set to "' + AUTOMATIC + '", '
                                 + 'the most suitable type is chosen automatically based on the parameter ' + PARAM_LOSS
                                 + '.')
        parser.add_argument(PARAM_PARALLEL_RULE_REFINEMENT, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('parallel_rule_refinement', AUTOMATIC,
                                                                           **kwargs),
                            help='Whether potential refinements of rules should be searched for in parallel or not. '
                                 + 'Must be one of ' + format_dict_keys(BOOSTING_PARALLEL_VALUES) + '. If set to "'
                                 + AUTOMATIC + '", the most suitable strategy is chosen automatically based on the '
                                 + 'parameter ' + PARAM_LOSS + '. For additional options refer to the documentation.')
        parser.add_argument(PARAM_PARALLEL_STATISTIC_UPDATE, type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('parallel_statistic_update', AUTOMATIC,
                                                                           **kwargs),
                            help='Whether the gradients and Hessians for different examples should be calculated in '
                                 + 'parallel or not. Must be one of ' + format_dict_keys(BOOSTING_PARALLEL_VALUES)
                                 + '. If set to "' + AUTOMATIC + '", the most suitable strategy is chosen '
                                 + 'automatically based on the parameter ' + PARAM_LOSS + '. For additional options '
                                 + 'refer to the documentation.')
        return self


    def build(self) -> ArgumentParser:
        return self.parser

    @staticmethod
    def __get_or_default(key: str, default_value, **kwargs):
        return kwargs[key] if key in kwargs else default_value
