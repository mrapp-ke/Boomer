#!/usr/bin/python

import argparse
import logging as log
from typing import List

import numpy as np

from args import log_level, optional_string, float_list, int_list, target_measure
from boomer.algorithm.model import DTYPE_FLOAT64
from boomer.baselines.problem_transformation import BRLearner, LPLearner, CCLearner
from boomer.baselines.xgboost import XGBoost
from boomer.bbc_cv import BbcCv, BbcCvAdapter, DefaultBbcCvObserver, DefaultBootstrapping


class BrBccCvAdapter(BbcCvAdapter):

    def __init__(self, data_dir: str, data_set: str, num_folds: int, model_dir: str):
        super().__init__(data_dir, data_set, num_folds, model_dir)

    def _store_predictions(self, model, test_indices, test_x, train_y, num_total_examples: int, num_labels: int,
                           predictions, configurations, current_fold, last_fold, num_folds):
        c = 0

        if len(predictions) > c:
            current_predictions = predictions[c]

        else:
            current_predictions = np.zeros((num_total_examples, num_labels), dtype=DTYPE_FLOAT64)
            predictions.append(current_predictions)
            current_config = self.configuration.copy()
            configurations.append(current_config)

        test_y = model.predict(test_x).toarray()

        if test_indices is None:
            current_predictions[:, :] = test_y
        else:
            current_predictions[test_indices, :] = test_y


class LpBccCvAdapter(BbcCvAdapter):

    def __init__(self, data_dir: str, data_set: str, num_folds: int, model_dir: str):
        super().__init__(data_dir, data_set, num_folds, model_dir)

    def _store_predictions(self, model, test_indices, test_x, train_y, num_total_examples: int, num_labels: int,
                           predictions, configurations, current_fold, last_fold, num_folds):
        c = 0

        if len(predictions) > c:
            current_predictions = predictions[c]
        else:
            current_predictions = np.empty((num_total_examples, num_labels), dtype=DTYPE_FLOAT64)
            predictions.append(current_predictions)
            current_config = self.configuration.copy()
            configurations.append(current_config)

        test_y = model.predict(test_x).toarray()

        if test_indices is None:
            current_predictions[:, :] = test_y
        else:
            current_predictions[test_indices, :] = test_y


class CcBccCvAdapter(BrBccCvAdapter):

    def __init__(self, data_dir: str, data_set: str, num_folds: int, model_dir: str):
        super().__init__(data_dir, data_set, num_folds, model_dir)

    def _store_predictions(self, model, test_indices, test_x, train_y, num_total_examples: int, num_labels: int,
                           predictions, configurations, current_fold: int, last_fold: int, num_folds: int):
        c = 0

        if len(predictions) > c:
            current_predictions = predictions[c]
        else:
            current_predictions = np.zeros((num_total_examples, num_labels), dtype=DTYPE_FLOAT64)
            predictions.append(current_predictions)
            current_config = self.configuration.copy()
            configurations.append(current_config)

        test_y = model.predict(test_x)

        if test_indices is None:
            current_predictions[:, :] = test_y
        else:
            current_predictions[test_indices, :] = test_y


def __create_configurations(cc: bool, objective_param: str, chain_order_values, arguments) -> List[dict]:
    learning_rate_values: List[float] = arguments.learning_rate
    reg_lambda_values: List[float] = arguments.reg_lambda
    result: List[dict] = []

    for chain_order in chain_order_values:
        for learning_rate in learning_rate_values:
            for reg_lambda in reg_lambda_values:
                configuration = {
                    'objective': objective_param,
                    'learning_rate': learning_rate,
                    'reg_lambda': reg_lambda,
                }

                if cc:
                    configuration.update({'chain_order': chain_order})

                result.append(configuration)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs BBC-CV using models that have been trained using CV')
    parser.add_argument('--log-level', type=log_level, default='info', help='The log level to be used')
    parser.add_argument('--random-state', type=int, default=1, help='The seed to be used by RNGs')
    parser.add_argument('--data-dir', type=str, help='The path of the directory where the data sets are located')
    parser.add_argument('--dataset', type=str, help='The name of the data set to be used')
    parser.add_argument('--folds', type=int, default=1, help='The total number of folds to be used by cross validation')
    parser.add_argument('--model-dir', type=str, help='The path of the directory where the models are stored')
    parser.add_argument('--output-dir', type=optional_string, default=None,
                        help='The path of the directory into which results should be written')
    parser.add_argument('--num-bootstraps', type=int, default=100,
                        help='The number of bootstrap iterations to be performed')
    parser.add_argument('--target-measure', type=target_measure, default='hamming-loss',
                        help='The target measure to be used for evaluating different configurations on the tuning set')
    parser.add_argument('--transformation-method', type=str, default='br',
                        help='The name of the problem transformation method to be used')
    parser.add_argument('--chain-order', type=int_list, default='1',
                        help='The values for the parameter \'chain-order\' as a comma-separated list')
    parser.add_argument('--learning-rate', type=float_list, default='1.0',
                        help='The values for the parameter \'learning-rate\' as a comma-separated list')
    parser.add_argument('--reg-lambda', type=float_list, default='1.0',
                        help='The values for the parameter \'reg-lambda\' as a comma-separated list')
    args = parser.parse_args()
    log.basicConfig(level=args.log_level)
    log.info('Configuration: %s', args)

    target_measure, target_measure_is_loss = args.target_measure
    transformation_method = args.transformation_method
    base_learner = XGBoost()

    if transformation_method == 'br':
        learner = BRLearner(model_dir=args.model_dir, base_learner=base_learner)
        objective = 'binary:logistic'
        bbc_cv_adapter = BrBccCvAdapter(data_dir=args.data_dir, data_set=args.dataset, num_folds=args.folds,
                                        model_dir=args.model_dir)
    elif transformation_method == 'lp':
        learner = LPLearner(model_dir=args.model_dir, base_learner=base_learner)
        objective = 'multi:softmax'
        bbc_cv_adapter = LpBccCvAdapter(data_dir=args.data_dir, data_set=args.dataset, num_folds=args.folds,
                                        model_dir=args.model_dir)
    elif transformation_method == 'cc':
        learner = CCLearner(model_dir=args.model_dir, base_learner=base_learner)
        objective = 'binary:logistic'
        bbc_cv_adapter = CcBccCvAdapter(data_dir=args.data_dir, data_set=args.dataset, num_folds=args.folds,
                                        model_dir=args.model_dir)
    else:
        raise ValueError('Invalid argument given: ' + str(transformation_method))

    base_configurations = __create_configurations(transformation_method == 'cc', objective, args.chain_order, args)
    bootstrapping = DefaultBootstrapping(num_bootstraps=args.num_bootstraps)
    bbc_cv = BbcCv(configurations=base_configurations, adapter=bbc_cv_adapter, bootstrapping=bootstrapping,
                   learner=learner)
    bbc_cv.random_state = args.random_state
    bbc_cv.store_predictions()
    bbc_cv.evaluate(observer=DefaultBbcCvObserver(output_dir=args.output_dir, target_measure=target_measure,
                                                  target_measure_is_loss=target_measure_is_loss))
