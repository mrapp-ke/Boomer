#!/usr/bin/python

from args import ArgumentParserBuilder
from mlrl.seco.seco_learners import SeparateAndConquerRuleLearner
from runnables import RuleLearnerRunnable


class SecoRunnable(RuleLearnerRunnable):

    def _create_learner(self, args):
        return SeparateAndConquerRuleLearner(random_state=args.random_state, feature_format=args.feature_format,
                                             label_format=args.label_format, max_rules=args.max_rules,
                                             time_limit=args.time_limit, loss=args.loss, heuristic=args.heuristic,
                                             pruning=args.pruning, label_sub_sampling=args.label_sub_sampling,
                                             instance_sub_sampling=args.instance_sub_sampling,
                                             feature_sub_sampling=args.feature_sub_sampling,
                                             holdout_set_size=args.holdout, feature_binning=args.feature_binning,
                                             head_refinement=args.head_refinement, min_coverage=args.min_coverage,
                                             max_conditions=args.max_conditions, lift_function=args.lift_function,
                                             max_head_refinements=args.max_head_refinements,
                                             num_threads_refinement=args.num_threads_refinement,
                                             num_threads_update=args.num_threads_update,
                                             num_threads_prediction=args.num_threads_prediction)


if __name__ == '__main__':
    parser = ArgumentParserBuilder(description='A multi-label classification experiment using Separate and Conquer') \
        .add_seco_learner_arguments() \
        .build()
    runnable = SecoRunnable()
    runnable.run(parser)
