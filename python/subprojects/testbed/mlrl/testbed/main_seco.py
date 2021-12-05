#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from argparse import ArgumentParser

from mlrl.common.options import BooleanOption
from mlrl.common.rule_learners import PRUNING_IREP, SAMPLING_WITHOUT_REPLACEMENT, HEAD_TYPE_SINGLE, PARALLEL_VALUES
from mlrl.common.strings import format_dict_keys, format_string_set
from mlrl.seco.seco_learners import SeCoRuleLearner, HEURISTIC_F_MEASURE, HEURISTIC_ACCURACY, LIFT_FUNCTION_PEAK, \
    INSTANCE_SAMPLING_VALUES as SECO_INSTANCE_SAMPLING_VALUES, HEAD_TYPE_VALUES as SECO_HEAD_TYPE_VALUES, \
    HEURISTIC_VALUES, LIFT_FUNCTION_VALUES, HEAD_TYPE_PARTIAL
from mlrl.testbed.args import add_rule_learner_arguments, get_or_default, optional_string, PARAM_INSTANCE_SAMPLING, \
    PARAM_HEAD_TYPE, PARAM_PARALLEL_RULE_REFINEMENT, PARAM_PARALLEL_STATISTIC_UPDATE, PARAM_PRUNING
from mlrl.testbed.runnables import RuleLearnerRunnable

PARAM_HEURISTIC = '--heuristic'

PARAM_PRUNING_HEURISTIC = '--pruning-heuristic'

PARAM_LIFT_FUNCTION = '--lift-function'


class SeCoRunnable(RuleLearnerRunnable):

    def _create_learner(self, args):
        return SeCoRuleLearner(random_state=args.random_state, feature_format=args.feature_format,
                               label_format=args.label_format, prediction_format=args.prediction_format,
                               max_rules=args.max_rules, time_limit=args.time_limit, heuristic=args.heuristic,
                               pruning_heuristic=args.pruning_heuristic, pruning=args.pruning,
                               label_sampling=args.label_sampling, instance_sampling=args.instance_sampling,
                               feature_sampling=args.feature_sampling, holdout=args.holdout,
                               feature_binning=args.feature_binning, head_type=args.head_type,
                               min_coverage=args.min_coverage, max_conditions=args.max_conditions,
                               lift_function=args.lift_function, max_head_refinements=args.max_head_refinements,
                               parallel_rule_refinement=args.parallel_rule_refinement,
                               parallel_statistic_update=args.parallel_statistic_update,
                               parallel_prediction=args.parallel_prediction)


def __add_arguments(parser: ArgumentParser, **kwargs):
    args = dict(kwargs)
    args[PARAM_PRUNING] = PRUNING_IREP
    add_rule_learner_arguments(parser, **args)
    parser.add_argument(PARAM_INSTANCE_SAMPLING, type=optional_string,
                        default=get_or_default(PARAM_INSTANCE_SAMPLING, SAMPLING_WITHOUT_REPLACEMENT, **kwargs),
                        help='The name of the strategy to be used for instance sampling. Must be one of '
                             + format_dict_keys(SECO_INSTANCE_SAMPLING_VALUES) + ' or "None", if no instance sampling '
                             + 'should be used. For additional options refer to the documentation.')
    parser.add_argument(PARAM_HEURISTIC, type=str,
                        default=get_or_default(PARAM_HEURISTIC, HEURISTIC_F_MEASURE, **kwargs),
                        help='The name of the heuristic to be used for learning rules. Must be one of '
                             + format_dict_keys(HEURISTIC_VALUES) + '. For additional options refer to the '
                             + 'documentation.')
    parser.add_argument(PARAM_PRUNING_HEURISTIC, type=str,
                        default=get_or_default(PARAM_PRUNING_HEURISTIC, HEURISTIC_ACCURACY, **kwargs),
                        help='The name of the heuristic to be used for pruning rules. Must be one of '
                             + format_dict_keys(HEURISTIC_VALUES) + '. For additional options refer to the '
                             + 'documentation.')
    parser.add_argument(PARAM_LIFT_FUNCTION, type=optional_string,
                        default=get_or_default(PARAM_LIFT_FUNCTION, LIFT_FUNCTION_PEAK, **kwargs),
                        help='The lift function to be used for the induction of multi-label rules. Must be one of '
                             + format_dict_keys(LIFT_FUNCTION_VALUES) + '. Does only have an effect if the parameter '
                             + PARAM_HEAD_TYPE + ' is set to "' + HEAD_TYPE_PARTIAL + '".')
    parser.add_argument(PARAM_HEAD_TYPE, type=str,
                        default=get_or_default(PARAM_HEAD_TYPE, HEAD_TYPE_SINGLE, **kwargs),
                        help='The type of the rule heads that should be used. Must be one of '
                             + format_string_set(SECO_HEAD_TYPE_VALUES) + '.')
    parser.add_argument(PARAM_PARALLEL_RULE_REFINEMENT, type=optional_string,
                        default=get_or_default(PARAM_PARALLEL_RULE_REFINEMENT, BooleanOption.TRUE.value, **kwargs),
                        help='Whether potential refinements of rules should be searched for in parallel or not. Must '
                             + 'be one of ' + format_dict_keys(PARALLEL_VALUES) + '. For additional options refer to '
                             + 'the documentation.')
    parser.add_argument(PARAM_PARALLEL_STATISTIC_UPDATE, type=optional_string,
                        default=get_or_default(PARAM_PARALLEL_STATISTIC_UPDATE, BooleanOption.FALSE.value, **kwargs),
                        help='Whether the confusion matrices for different examples should be calculated in parallel '
                             + 'or not. Must be one of ' + format_dict_keys(PARALLEL_VALUES) + '. For additional '
                             + 'options refer to the documentation')


def main():
    parser = ArgumentParser(description='Allows to run experiments using the Separate-and-Conquer algorithm')
    __add_arguments(parser)
    runnable = SeCoRunnable()
    runnable.run(parser)


if __name__ == '__main__':
    main()
