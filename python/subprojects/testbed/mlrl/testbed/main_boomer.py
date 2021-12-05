#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from argparse import ArgumentParser

from mlrl.common.options import BooleanOption
from mlrl.common.rule_learners import AUTOMATIC, SAMPLING_WITHOUT_REPLACEMENT, INSTANCE_SAMPLING_VALUES
from mlrl.common.strings import format_enum_values, format_dict_keys, format_string_set
from mlrl.testbed.args import add_rule_learner_arguments, get_or_default, optional_string, PARAM_INSTANCE_SAMPLING, \
    PARAM_PARTITION_SAMPLING, PARAM_HEAD_TYPE, PARAM_PARALLEL_RULE_REFINEMENT, PARAM_PARALLEL_STATISTIC_UPDATE, \
    PARAM_MAX_RULES, PARAM_FEATURE_SAMPLING
from mlrl.testbed.runnables import RuleLearnerRunnable

from mlrl.boosting.boosting_learners import Boomer, LOSS_LOGISTIC_LABEL_WISE, HEAD_TYPE_VALUES, EARLY_STOPPING_VALUES, \
    LABEL_BINNING_VALUES, LOSS_VALUES, PREDICTOR_VALUES, PARALLEL_VALUES

PARAM_DEFAULT_RULE = '--default-rule'

PARAM_RECALCULATE_PREDICTIONS = '--recalculate-predictions'

PARAM_EARLY_STOPPING = '--early-stopping'

PARAM_LABEL_BINNING = '--label-binning'

PARAM_LOSS = '--loss'

PARAM_SHRINKAGE = '--shrinkage'

PARAM_PREDICTOR = '--predictor'

PARAM_L1_REGULARIZATION_WEIGHT = '--l1-regularization-weight'

PARAM_L2_REGULARIZATION_WEIGHT = '--l2-regularization-weight'


class BoomerRunnable(RuleLearnerRunnable):

    def _create_learner(self, args):
        return Boomer(random_state=args.random_state, feature_format=args.feature_format,
                      label_format=args.label_format, prediction_format=args.prediction_format,
                      max_rules=args.max_rules, default_rule=args.default_rule, time_limit=args.time_limit,
                      early_stopping=args.early_stopping, loss=args.loss, predictor=args.predictor,
                      pruning=args.pruning, label_sampling=args.label_sampling,
                      instance_sampling=args.instance_sampling, recalculate_predictions=args.recalculate_predictions,
                      shrinkage=args.shrinkage, feature_sampling=args.feature_sampling, holdout=args.holdout,
                      feature_binning=args.feature_binning, label_binning=args.label_binning, head_type=args.head_type,
                      l1_regularization_weight=args.l1_regularization_weight,
                      l2_regularization_weight=args.l2_regularization_weight, min_coverage=args.min_coverage,
                      max_conditions=args.max_conditions, max_head_refinements=args.max_head_refinements,
                      parallel_rule_refinement=args.parallel_rule_refinement,
                      parallel_statistic_update=args.parallel_statistic_update,
                      parallel_prediction=args.parallel_prediction)


def __add_arguments(parser: ArgumentParser, **kwargs):
    args = dict(kwargs)
    args[PARAM_MAX_RULES] = 1000
    args[PARAM_FEATURE_SAMPLING] = SAMPLING_WITHOUT_REPLACEMENT
    add_rule_learner_arguments(parser, **args)
    parser.add_argument(PARAM_DEFAULT_RULE, type=optional_string,
                        default=get_or_default(PARAM_DEFAULT_RULE, BooleanOption.TRUE.value, **kwargs),
                        help='Whether the first rule should be a default rule or not. Must be one of '
                             + format_enum_values(BooleanOption))
    parser.add_argument(PARAM_RECALCULATE_PREDICTIONS, type=optional_string,
                        default=get_or_default(PARAM_RECALCULATE_PREDICTIONS, BooleanOption.TRUE.value, **kwargs),
                        help='Whether the predictions of rules should be recalculated on the entire training data, if '
                             + 'the parameter ' + PARAM_INSTANCE_SAMPLING + ' is not set to None, or not. Must be one '
                             + 'of ' + format_enum_values(BooleanOption) + '.')
    parser.add_argument(PARAM_EARLY_STOPPING, type=optional_string,
                        default=get_or_default(PARAM_EARLY_STOPPING, None, **kwargs),
                        help='The name of the strategy to be used for early stopping. Must be one of '
                             + format_dict_keys(EARLY_STOPPING_VALUES) + ' or "None", if no early stopping should be '
                             + 'used. Does only have an effect if the parameter ' + PARAM_PARTITION_SAMPLING + ' is '
                             + 'not set to "None". For additional options refer to the documentation.')
    parser.add_argument(PARAM_INSTANCE_SAMPLING, type=optional_string,
                        default=get_or_default(PARAM_INSTANCE_SAMPLING, None, **kwargs),
                        help='The name of the strategy to be used for instance sampling. Must be one of'
                             + format_dict_keys(INSTANCE_SAMPLING_VALUES) + ' or "None", if no instance sampling '
                             + 'should be used. For additional options refer to the documentation.')
    parser.add_argument(PARAM_LABEL_BINNING, type=optional_string,
                        default=get_or_default(PARAM_LABEL_BINNING, AUTOMATIC, **kwargs),
                        help='The name of the strategy to be used for gradient-based label binning (GBLB). Must be one '
                             + 'of ' + format_dict_keys(LABEL_BINNING_VALUES) + ' or "None", if no label binning '
                             + 'should be used. If set to "' + AUTOMATIC + '", the most suitable strategy is chosen '
                             + 'automatically based on the parameters ' + PARAM_LOSS + ' and ' + PARAM_HEAD_TYPE + '. '
                             + 'For additional options refer to the documentation.')
    parser.add_argument(PARAM_SHRINKAGE, type=float,
                        default=get_or_default(PARAM_SHRINKAGE, 0.3, **kwargs),
                        help='The shrinkage parameter, a.k.a. the learning rate, to be used. Must be in (0, 1].')
    parser.add_argument(PARAM_LOSS, type=str,
                        default=get_or_default(PARAM_LOSS, LOSS_LOGISTIC_LABEL_WISE, **kwargs),
                        help='The name of the loss function to be minimized during training. Must be one of '
                             + format_string_set(LOSS_VALUES) + '.')
    parser.add_argument(PARAM_PREDICTOR, type=str,
                        default=get_or_default(PARAM_PREDICTOR, AUTOMATIC, **kwargs),
                        help='The name of the strategy to be used for making predictions. Must be one of '
                             + format_string_set(PREDICTOR_VALUES) + '. If set to "' + AUTOMATIC + '", the most '
                             + 'suitable strategy is chosen automatically based on the parameter ' + PARAM_LOSS
                             + '.')
    parser.add_argument(PARAM_L1_REGULARIZATION_WEIGHT, type=float,
                        default=get_or_default(PARAM_L1_REGULARIZATION_WEIGHT, 0.0, **kwargs),
                        help='The weight of the L1 regularization. Must be at least 0.')
    parser.add_argument(PARAM_L2_REGULARIZATION_WEIGHT, type=float,
                        default=get_or_default(PARAM_L2_REGULARIZATION_WEIGHT, 1.0, **kwargs),
                        help='The weight of the L2 regularization. Must be at least 0.')
    parser.add_argument(PARAM_HEAD_TYPE, type=str,
                        default=get_or_default(PARAM_HEAD_TYPE, AUTOMATIC, **kwargs),
                        help='The type of the rule heads that should be used. Must be one of '
                             + format_string_set(HEAD_TYPE_VALUES) + '. If set to "' + AUTOMATIC + '", the most '
                             + 'suitable type is chosen automatically based on the parameter ' + PARAM_LOSS + '.')
    parser.add_argument(PARAM_PARALLEL_RULE_REFINEMENT, type=optional_string,
                        default=get_or_default(PARAM_PARALLEL_RULE_REFINEMENT, AUTOMATIC, **kwargs),
                        help='Whether potential refinements of rules should be searched for in parallel or not. Must '
                             + 'be one of ' + format_dict_keys(PARALLEL_VALUES) + '. If set to "' + AUTOMATIC + '", '
                             + 'the most suitable strategy is chosen automatically based on the parameter ' + PARAM_LOSS
                             + '. For additional options refer to the documentation.')
    parser.add_argument(PARAM_PARALLEL_STATISTIC_UPDATE, type=optional_string,
                        default=get_or_default(PARAM_PARALLEL_STATISTIC_UPDATE, AUTOMATIC, **kwargs),
                        help='Whether the gradients and Hessians for different examples should be calculated in '
                             + 'parallel or not. Must be one of ' + format_dict_keys(PARALLEL_VALUES) + '. If set to "'
                             + AUTOMATIC + '", the most suitable strategy is chosen automatically based on the '
                             + 'parameter ' + PARAM_LOSS + '. For additional options refer to the documentation.')


def main():
    parser = ArgumentParser(description='Allows to run experiments using the BOOMER algorithm')
    __add_arguments(parser)
    runnable = BoomerRunnable()
    runnable.run(parser)


if __name__ == '__main__':
    main()
