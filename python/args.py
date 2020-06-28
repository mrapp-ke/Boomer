#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides functions for parsing command line arguments.
"""
import logging as log
from argparse import ArgumentParser

import sklearn.metrics as metrics

from boomer.boosting.boosting_learners import LOSS_LABEL_WISE_LOGISTIC
from boomer.common.rule_learners import INSTANCE_SUB_SAMPLING_BAGGING, FEATURE_SUB_SAMPLING_RANDOM
from boomer.seco.seco_learners import HEURISTIC_PRECISION, LIFT_FUNCTION_PEAK, AVERAGING_LABEL_WISE


def log_level(s):
    s = s.lower()
    if s == 'debug':
        return log.DEBUG
    elif s == 'info':
        return log.INFO
    elif s == 'warn' or s == 'warning':
        return log.WARN
    elif s == 'error':
        return log.ERROR
    elif s == 'critical' or s == 'fatal':
        return log.CRITICAL
    elif s == 'notset':
        return log.NOTSET
    raise ValueError('Invalid argument given for parameter \'--log-level\': ' + str(s))


def current_fold_string(s):
    n = int(s)
    if n > 0:
        return n - 1
    elif n == -1:
        return -1
    raise ValueError('Invalid argument given for parameter \'--current-fold\': ' + str(n))


def target_measure(s):
    if s == 'hamming-loss':
        return metrics.hamming_loss, True
    elif s == 'subset-0-1-loss':
        return metrics.accuracy_score, False
    raise ValueError('Invalid argument given for parameter \'--target-measure\': ' + str(s))


def boolean_string(s):
    s = s.lower()

    if s == 'false':
        return False
    if s == 'true':
        return True
    raise ValueError('Invalid boolean argument given: ' + str(s))


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
        parser.add_argument('--log-level', type=log_level,
                            default=ArgumentParserBuilder.__get_or_default('log_level', 'info', **kwargs),
                            help='The log level to be used')
        self.parser = parser

    def add_random_state_argument(self, **kwargs) -> 'ArgumentParserBuilder':
        self.parser.add_argument('--random-state', type=int,
                                 default=ArgumentParserBuilder.__get_or_default('random_state', 1, **kwargs),
                                 help='The seed to be used by RNGs')
        return self

    def add_learner_arguments(self, **kwargs) -> 'ArgumentParserBuilder':
        parser = self.parser
        parser.add_argument('--data-dir', type=str, help='The path of the directory where the data sets are located')
        parser.add_argument('--output-dir', type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('output_dir', None, **kwargs),
                            help='The path of the directory into which results should be written')
        parser.add_argument('--model-dir', type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('model_dir', None, **kwargs),
                            help='The path of the directory where models should be saved')
        parser.add_argument('--dataset', type=str, help='The name of the data set to be used')
        parser.add_argument('--one-hot-encoding', type=boolean_string,
                            default=ArgumentParserBuilder.__get_or_default('one_hot_encoding', False, **kwargs),
                            help='True, if one-hot-encoding should be used, False otherwise')
        parser.add_argument('--folds', type=int, default=1, help='Total number of folds to be used by cross validation')
        parser.add_argument('--current-fold', type=current_fold_string,
                            default=ArgumentParserBuilder.__get_or_default('current_fold', -1, **kwargs),
                            help='The cross validation fold to be performed')
        parser.add_argument('--store-predictions', type=boolean_string,
                            default=ArgumentParserBuilder.__get_or_default('store_predictions', False, **kwargs),
                            help='True, if the predictions should be stored as CSV files, False otherwise')
        parser.add_argument('--parameter-dir', type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('parameter_dir', None, **kwargs),
                            help='The path of the directory, parameter settings should be loaded from')
        parser.add_argument('--evaluate-training-data', type=boolean_string,
                            default=ArgumentParserBuilder.__get_or_default('evaluate_training_data', False, **kwargs),
                            help='True, if the models should be evaluated on the training data, False otherwise')
        return self

    def add_rule_learner_arguments(self, loss: str, **kwargs) -> 'ArgumentParserBuilder':
        self.add_learner_arguments(**kwargs)
        self.add_random_state_argument(**kwargs)
        parser = self.parser
        parser.add_argument('--max-rules', type=int,
                            default=ArgumentParserBuilder.__get_or_default('max_rules', 500, **kwargs),
                            help='The maximum number of rules to be induced or -1')
        parser.add_argument('--time-limit', type=int,
                            default=ArgumentParserBuilder.__get_or_default('time_limit', -1, **kwargs),
                            help='The duration in seconds after which the induction of rules should be canceled or -1')
        parser.add_argument('--label-sub-sampling', type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('label_sub_sampling', None, **kwargs),
                            help='The name of the strategy to be used for label sub-sampling or None')
        parser.add_argument('--instance-sub-sampling', type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('instance_sub_sampling', None, **kwargs),
                            help='The name of the strategy to be used for instance sub-sampling or None')
        parser.add_argument('--feature-sub-sampling', type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('feature_sub_sampling', None, **kwargs),
                            help='The name of the strategy to be used for feature sub-sampling or None')
        parser.add_argument('--loss', type=str, default=loss, help='The name of the loss function to be used')
        parser.add_argument('--head-refinement', type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('head_refinement', None, **kwargs),
                            help='The name of the strategy to be used for finding the heads of rules')
        parser.add_argument('--pruning', type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('pruning', None, **kwargs),
                            help='The name of the strategy to be used for pruning or None')
        parser.add_argument('--min-coverage', type=int,
                            default=ArgumentParserBuilder.__get_or_default('min_coverage', 1, **kwargs),
                            help='The minimum number of training examples that must be covered by a rule')
        parser.add_argument('--max-conditions', type=int,
                            default=ArgumentParserBuilder.__get_or_default('max_conditions', -1, **kwargs),
                            help='The maximum number of conditions to be included in a rule\'s body or -1')
        parser.add_argument('--max-head-refinements', type=int,
                            default=ArgumentParserBuilder.__get_or_default('max_head_refinements', -1, **kwargs),
                            help='The maximum number of times the head of a rule may be refined or -1')
        parser.add_argument('--print-rules', type=boolean_string,
                            default=ArgumentParserBuilder.__get_or_default('print_rules', False, **kwargs),
                            help='True, if the induced rules should be printed on the console, False otherwise')
        parser.add_argument('--store-rules', type=boolean_string,
                            default=ArgumentParserBuilder.__get_or_default('store_rules', False, **kwargs),
                            help='True, if the induced rules should be stored in TXT files, False otherwise')
        return self

    def add_boosting_learner_arguments(self, **kwargs) -> 'ArgumentParserBuilder':
        self.add_rule_learner_arguments(LOSS_LABEL_WISE_LOGISTIC, max_rules=1000,
                                        instance_sub_sampling=INSTANCE_SUB_SAMPLING_BAGGING,
                                        feature_sub_sampling=FEATURE_SUB_SAMPLING_RANDOM, **kwargs)
        parser = self.parser
        parser.add_argument('--l2-regularization-weight', type=float,
                            default=ArgumentParserBuilder.__get_or_default('l2_regularization_weight', 1.0, **kwargs),
                            help='The weight of the L2 regularization to be used')
        parser.add_argument('--shrinkage', type=float,
                            default=ArgumentParserBuilder.__get_or_default('shrinkage', 0.3, **kwargs),
                            help='The shrinkage parameter to be used')
        return self

    def add_seco_learner_arguments(self, **kwargs) -> 'ArgumentParserBuilder':
        self.add_rule_learner_arguments(AVERAGING_LABEL_WISE, print_rules=True, **kwargs)
        parser = self.parser
        parser.add_argument('--heuristic', type=str,
                            default=ArgumentParserBuilder.__get_or_default('heuristic', HEURISTIC_PRECISION, **kwargs),
                            help='The name of the heuristic to be used')
        parser.add_argument('--lift-function', type=optional_string,
                            default=ArgumentParserBuilder.__get_or_default('lift_function', LIFT_FUNCTION_PEAK,
                                                                           **kwargs),
                            help='The lift function to be used')
        return self

    def build(self) -> ArgumentParser:
        return self.parser

    @staticmethod
    def __get_or_default(key: str, default_value, **kwargs):
        return kwargs[key] if key in kwargs else default_value
