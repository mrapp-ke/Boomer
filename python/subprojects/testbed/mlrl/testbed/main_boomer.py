#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from argparse import ArgumentParser

from mlrl.common.rule_learners import AUTOMATIC, RULE_MODEL_ASSEMBLAGE_VALUES
from mlrl.common.strings import format_dict_keys, format_string_set
from mlrl.testbed.args import add_rule_learner_arguments, PARAM_HEAD_TYPE, PARAM_PARALLEL_RULE_REFINEMENT, \
    PARAM_PARALLEL_STATISTIC_UPDATE
from mlrl.testbed.runnables import RuleLearnerRunnable

from mlrl.boosting.boosting_learners import Boomer, HEAD_TYPE_VALUES, EARLY_STOPPING_VALUES, LABEL_BINNING_VALUES, \
    LOSS_VALUES, PREDICTOR_VALUES, PARALLEL_VALUES, FEATURE_BINNING_VALUES

PARAM_RULE_MODEL_ASSEMBLAGE = '--rule-model-assemblage'

PARAM_EARLY_STOPPING = '--early-stopping'

PARAM_FEATURE_BINNING = '--feature-binning'

PARAM_LABEL_BINNING = '--label-binning'

PARAM_LOSS = '--loss'

PARAM_SHRINKAGE = '--shrinkage'

PARAM_PREDICTOR = '--predictor'

PARAM_L1_REGULARIZATION_WEIGHT = '--l1-regularization-weight'

PARAM_L2_REGULARIZATION_WEIGHT = '--l2-regularization-weight'


class BoomerRunnable(RuleLearnerRunnable):

    def _create_learner(self, args):
        return Boomer(random_state=args.random_state,
                      feature_format=args.feature_format,
                      label_format=args.label_format,
                      prediction_format=args.prediction_format,
                      rule_model_assemblage=args.rule_model_assemblage,
                      rule_induction=args.rule_induction,
                      max_rules=args.max_rules,
                      time_limit=args.time_limit,
                      early_stopping=args.early_stopping,
                      loss=args.loss,
                      predictor=args.predictor,
                      pruning=args.pruning,
                      label_sampling=args.label_sampling,
                      instance_sampling=args.instance_sampling,
                      shrinkage=args.shrinkage,
                      feature_sampling=args.feature_sampling,
                      holdout=args.holdout,
                      feature_binning=args.feature_binning,
                      label_binning=args.label_binning,
                      head_type=args.head_type,
                      l1_regularization_weight=args.l1_regularization_weight,
                      l2_regularization_weight=args.l2_regularization_weight,
                      parallel_rule_refinement=args.parallel_rule_refinement,
                      parallel_statistic_update=args.parallel_statistic_update,
                      parallel_prediction=args.parallel_prediction)


def __add_arguments(parser: ArgumentParser):
    add_rule_learner_arguments(parser)
    parser.add_argument(PARAM_RULE_MODEL_ASSEMBLAGE, type=str,
                        help='The name of the algorithm to be used for the induction of several rule. Must be one of '
                             + format_string_set(RULE_MODEL_ASSEMBLAGE_VALUES) + '. For additional options refer to '
                             + 'the documentation.')
    parser.add_argument(PARAM_EARLY_STOPPING, type=str,
                        help='The name of the strategy to be used for early stopping. Must be one of '
                             + format_dict_keys(EARLY_STOPPING_VALUES) + '. For additional options refer to the '
                             + 'documentation.')
    parser.add_argument(PARAM_FEATURE_BINNING, type=str,
                        help='The name of the strategy to be used for feature binning. Must be one of '
                             + format_dict_keys(FEATURE_BINNING_VALUES) + '. If set to "' + AUTOMATIC + '", the most '
                             + 'suitable strategy is chosen automatically based on the characteristics of the feature '
                             + 'matrix. For additional options refer to the documentation.')
    parser.add_argument(PARAM_LABEL_BINNING, type=str,
                        help='The name of the strategy to be used for gradient-based label binning (GBLB). Must be one '
                             + 'of ' + format_dict_keys(LABEL_BINNING_VALUES) + '. If set to "' + AUTOMATIC + '", the '
                             + 'most suitable strategy is chosen automatically based on the parameters ' + PARAM_LOSS
                             + ' and ' + PARAM_HEAD_TYPE + '. For additional options refer to the documentation.')
    parser.add_argument(PARAM_SHRINKAGE, type=float,
                        help='The shrinkage parameter, a.k.a. the learning rate, to be used. Must be in (0, 1].')
    parser.add_argument(PARAM_LOSS, type=str,
                        help='The name of the loss function to be minimized during training. Must be one of '
                             + format_string_set(LOSS_VALUES) + '.')
    parser.add_argument(PARAM_PREDICTOR, type=str,
                        help='The name of the strategy to be used for making predictions. Must be one of '
                             + format_string_set(PREDICTOR_VALUES) + '. If set to "' + AUTOMATIC + '", the most '
                             + 'suitable strategy is chosen automatically based on the parameter ' + PARAM_LOSS + '.')
    parser.add_argument(PARAM_L1_REGULARIZATION_WEIGHT, type=float,
                        help='The weight of the L1 regularization. Must be at least 0.')
    parser.add_argument(PARAM_L2_REGULARIZATION_WEIGHT, type=float,
                        help='The weight of the L2 regularization. Must be at least 0.')
    parser.add_argument(PARAM_HEAD_TYPE, type=str,
                        help='The type of the rule heads that should be used. Must be one of '
                             + format_string_set(HEAD_TYPE_VALUES) + '. If set to "' + AUTOMATIC + '", the most '
                             + 'suitable type is chosen automatically based on the parameter ' + PARAM_LOSS + '.')
    parser.add_argument(PARAM_PARALLEL_RULE_REFINEMENT, type=str,
                        help='Whether potential refinements of rules should be searched for in parallel or not. Must '
                             + 'be one of ' + format_dict_keys(PARALLEL_VALUES) + '. If set to "' + AUTOMATIC
                             + '", the most suitable strategy is chosen automatically based on the parameter '
                             + PARAM_LOSS + '. For additional options refer to the documentation.')
    parser.add_argument(PARAM_PARALLEL_STATISTIC_UPDATE, type=str,
                        help='Whether the gradients and Hessians for different examples should be calculated in '
                             + 'parallel or not. Must be one of ' + format_dict_keys(PARALLEL_VALUES) + '. If set '
                             + 'to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically based on '
                             + 'the parameter ' + PARAM_LOSS + '. For additional options refer to the documentation.')


def main():
    parser = ArgumentParser(description='Allows to run experiments using the BOOMER algorithm')
    __add_arguments(parser)
    runnable = BoomerRunnable()
    runnable.run(parser)


if __name__ == '__main__':
    main()
