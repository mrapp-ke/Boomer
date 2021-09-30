#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from args import ArgumentParserBuilder
from mlrl.boosting.boosting_learners import Boomer
from runnables import RuleLearnerRunnable


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
                      l2_regularization_weight=args.l2_regularization_weight, min_coverage=args.min_coverage,
                      max_conditions=args.max_conditions, max_head_refinements=args.max_head_refinements,
                      parallel_rule_refinement=args.parallel_rule_refinement,
                      parallel_statistic_update=args.parallel_statistic_update,
                      parallel_prediction=args.parallel_prediction)


if __name__ == '__main__':
    parser = ArgumentParserBuilder(description='Allows to run experiments using the BOOMER algorithm') \
        .add_boosting_learner_arguments() \
        .build()
    runnable = BoomerRunnable()
    runnable.run(parser)
