#!/usr/bin/python

import argparse
import logging as log

from args import optional_string, log_level, boolean_string, current_fold_string
from boomer.algorithm.rule_learners import Boomer
from boomer.evaluation import ClassificationEvaluation, EvaluationLogOutput, EvaluationCsvOutput
from boomer.experiments import Experiment
from boomer.parameters import ParameterCsvInput


def configure_argument_parser(p: argparse.ArgumentParser):
    p.add_argument('--log-level', type=log_level, default='info', help='The log level to be used')
    p.add_argument('--data-dir', type=str, help='The path of the directory where the data sets are located')
    p.add_argument('--output-dir', type=optional_string, default=None,
                   help='The path of the directory into which results should be written')
    p.add_argument('--model-dir', type=optional_string, default=None,
                   help='The path of the directory where models should be saved')
    p.add_argument('--dataset', type=str, help='The name of the data set to be used')
    p.add_argument('--folds', type=int, default=1, help='Total number of folds to be used by cross validation')
    p.add_argument('--current-fold', type=current_fold_string, default=-1,
                   help='The cross validation fold to be performed')
    p.add_argument('--num-rules', type=int, default=500, help='The number of rules to be induced or -1')
    p.add_argument('--time-limit', type=int, default=-1,
                   help='The duration in seconds after which the induction of rules should be canceled or -1')
    p.add_argument('--label-sub-sampling', type=int, default=-1,
                   help='The number of samples to be used for label sub-sampling or -1')
    p.add_argument('--instance-sub-sampling', type=optional_string, default=None,
                   help='The name of the strategy to be used for instance sub-sampling or None')
    p.add_argument('--feature-sub-sampling', type=optional_string, default=None,
                   help='The name of the strategy to be used for feature sub-sampling or None')
    p.add_argument('--pruning', type=optional_string, default=None,
                   help='The name of the strategy to be used for pruning or None')
    p.add_argument('--loss', type=str, default='macro-squared-error-loss',
                   help='The name of the loss function to be used')
    p.add_argument('--l2-regularization-weight', type=float, default=0,
                   help='The weight of the L2 regularization to be used')
    p.add_argument('--head-refinement', type=optional_string, default=None,
                   help='The name of the strategy to be used for finding the heads of rules')
    p.add_argument('--shrinkage', type=float, default=1.0, help='The shrinkage parameter to be used')


def create_learner(params) -> Boomer:
    return Boomer(model_dir=params.model_dir, num_rules=params.num_rules, time_limit=params.time_limit,
                  loss=params.loss, pruning=params.pruning, label_sub_sampling=params.label_sub_sampling,
                  instance_sub_sampling=params.instance_sub_sampling, shrinkage=params.shrinkage,
                  feature_sub_sampling=params.feature_sub_sampling, head_refinement=params.head_refinement,
                  l2_regularization_weight=params.l2_regularization_weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An multi-label classification experiment using BOOMER')
    configure_argument_parser(parser)
    parser.add_argument('--random-state', type=int, default=1, help='The seed to be used by RNGs')
    parser.add_argument('--store-predictions', type=boolean_string, default=False,
                        help='True, if the predictions should be stored as CSV files, False otherwise')
    parser.add_argument('--parameter-dir', type=optional_string, default=None,
                        help='The path of the directory, parameter settings should be loaded from')
    args = parser.parse_args()
    log.basicConfig(level=args.log_level)
    log.info('Configuration: %s', args)

    parameter_input = None if args.parameter_dir is None else ParameterCsvInput(input_dir=args.parameter_dir)
    evaluation_outputs = [EvaluationLogOutput()]

    if args.output_dir is not None:
        evaluation_outputs.append(EvaluationCsvOutput(output_dir=args.output_dir,
                                                      output_predictions=args.store_predictions,
                                                      clear_dir=args.current_fold == -1))

    learner = create_learner(args)
    parameter_input = parameter_input
    evaluation = ClassificationEvaluation(EvaluationLogOutput(), *evaluation_outputs)
    experiment = Experiment(learner, evaluation, data_dir=args.data_dir, data_set=args.dataset, num_folds=args.folds,
                            current_fold=args.current_fold, parameter_input=parameter_input)
    experiment.run()
