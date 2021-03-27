#!/usr/bin/python

from args import ArgumentParserBuilder
from mlrl.boosting.boosting_learners import Boomer
from runnables import RuleLearnerRunnable


class BoomerRunnable(RuleLearnerRunnable):

    def _create_learner(self, args):
        return Boomer(random_state=args.random_state, feature_format=args.feature_format,
                      label_format=args.label_format, max_rules=args.max_rules, time_limit=args.time_limit,
                      early_stopping=args.early_stopping, loss=args.loss, predictor=args.predictor,
                      pruning=args.pruning, label_sub_sampling=args.label_sub_sampling,
                      instance_sub_sampling=args.instance_sub_sampling, shrinkage=args.shrinkage,
                      feature_sub_sampling=args.feature_sub_sampling, holdout_set_size=args.holdout,
                      feature_binning=args.feature_binning, head_refinement=args.head_refinement,
                      l2_regularization_weight=args.l2_regularization_weight, min_coverage=args.min_coverage,
                      max_conditions=args.max_conditions, max_head_refinements=args.max_head_refinements,
                      num_threads_refinement=args.num_threads_refinement, num_threads_update=args.num_threads_update,
                      num_threads_prediction=args.num_threads_prediction)


if __name__ == '__main__':
    parser = ArgumentParserBuilder(description='A multi-label classification experiment using BOOMER') \
        .add_boosting_learner_arguments() \
        .build()
    runnable = BoomerRunnable()
    runnable.run(parser)
