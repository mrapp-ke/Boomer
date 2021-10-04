#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from args import ArgumentParserBuilder
from mlrl.seco.seco_learners import SeCoRuleLearner
from runnables import RuleLearnerRunnable


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


if __name__ == '__main__':
    parser = ArgumentParserBuilder(description='Allows to run experiments using the Separate-and-Conquer algorithm') \
        .add_seco_learner_arguments() \
        .build()
    runnable = SeCoRunnable()
    runnable.run(parser)
