#!/usr/bin/python

import argparse
import logging as log
from typing import List

import numpy as np
import scipy.stats as stats
from skmultilearn.problem_transform import LabelPowerset

from args import optional_string, log_level, string_list, float_list, int_list, target_measure, boolean_string
from boomer.algorithm.model import DTYPE_FLOAT64
from boomer.algorithm.rule_learners import Boomer
from boomer.bbc_cv import BbcCv, BbcCvAdapter, BbcCvObserver, DefaultBbcCvObserver, DefaultBootstrapping
from boomer.evaluation import ClassificationEvaluation, EvaluationLogOutput, EvaluationCsvOutput


class BoomerBccCvAdapter(BbcCvAdapter):

    def __init__(self, data_dir: str, data_set: str, num_folds: int, model_dir: str, min_rules: int, max_rules: int,
                 step_size_rules: int, subset_correction: bool):
        super().__init__(data_dir, data_set, num_folds, model_dir)
        self.min_rules = min_rules
        self.max_rules = max_rules
        self.step_size_rules = step_size_rules
        self.subset_correction = subset_correction

    def _store_predictions(self, model, test_indices, test_x, train_y, num_total_examples: int, num_labels: int,
                           predictions, configurations, current_fold, last_fold, num_folds):
        num_rules = len(model)
        c = 0

        if len(predictions) > c:
            current_predictions = predictions[c]
            current_config = configurations[c]
        else:
            current_predictions = np.zeros((num_total_examples, num_labels), dtype=DTYPE_FLOAT64)
            predictions.append(current_predictions)
            current_config = self.configuration.copy()
            configurations.append(current_config)

        # Store predictions...
        min_rules = self.min_rules
        min_rules = max(min_rules, 1) if min_rules != -1 else 1
        max_rules = self.max_rules
        max_rules = min(num_rules, max_rules) if max_rules != -1 else num_rules
        step_size = min(max(1, self.step_size_rules), max_rules)

        for n in range(max_rules):
            rule = model.pop(0)

            if np.isnan(np.asarray(rule.head.scores)).any():
                log.error("There's something wrong with this rule")
                raise ArithmeticError()
            else:
                if test_indices is None:
                    rule.predict(test_x, current_predictions)
                else:
                    masked_predictions = current_predictions[test_indices, :]
                    rule.predict(test_x, masked_predictions)
                    current_predictions[test_indices, :] = masked_predictions

            current_config['num_rules'] = (n + 1)

            if min_rules <= (n + 1) <= max_rules - 1 and (n + 1) % step_size == 0:
                c += 1

                if len(predictions) > c:
                    old_predictions = current_predictions
                    current_predictions = predictions[c]

                    if test_indices is None:
                        current_predictions[:, :] = old_predictions[:, :]
                    else:
                        current_predictions[test_indices] = old_predictions[test_indices]

                    current_config = configurations[c]
                else:
                    current_predictions = current_predictions.copy()
                    predictions.append(current_predictions)
                    current_config = current_config.copy()
                    configurations.append(current_config)

        if self.subset_correction:
            lp = LabelPowerset()
            lp.transform(train_y)
            num_labels = lp._label_count
            reverse_combinations = lp.reverse_combinations_
            num_label_sets = len(reverse_combinations)
            unique_label_sets = np.zeros((num_label_sets, num_labels))

            for n in range(num_label_sets):
                unique_label_sets[n, reverse_combinations[n]] = 1

            unique_label_sets = -np.where(unique_label_sets > 0, 1, -1)

            for c in range(len(predictions)):
                current_predictions = predictions[c]

                if test_indices is None:
                    raise NotImplementedError()
                else:
                    masked_predictions = current_predictions[test_indices, :]
                    num_predictions = masked_predictions.shape[0]
                    mapped_predictions = np.zeros(masked_predictions.shape)

                    for r in range(num_predictions):
                        pred = masked_predictions[r, :]
                        distances = np.log(1 + np.sum(np.exp(unique_label_sets * pred), axis=1))
                        index = np.argmin(distances)
                        mapped_predictions[r, reverse_combinations[index]] = 1

                    current_predictions[test_indices, :] = mapped_predictions


class TuningEvaluationBbcCvObserver(BbcCvObserver):

    def __init__(self, measure):
        """
        :param measure: The target measure to be used for parameter tuning
        """
        self.target_measure = measure
        self.evaluation_scores_tuning = None

    def evaluate(self, configurations: List[dict], ground_truth_tuning: np.ndarray, predictions_tuning: np.ndarray,
                 ground_truth_test: np.ndarray, predictions_test: np.ndarray, current_bootstrap: int,
                 num_bootstraps: int):
        measure = self.target_measure
        evaluation_scores_tuning = self.evaluation_scores_tuning
        num_configurations = len(configurations)

        if evaluation_scores_tuning is None:
            evaluation_scores_tuning = np.empty((num_configurations, num_bootstraps), dtype=float)
            self.evaluation_scores_tuning = evaluation_scores_tuning

        for k in range(num_configurations):
            predictions = predictions_tuning[:, k, :]
            evaluation_scores_tuning[k, current_bootstrap] = measure(ground_truth_tuning, predictions)


class BestConfigBbcCvObserver(BbcCvObserver):

    def __init__(self, measure, measure_is_loss: bool, best_configuration: dict = None, output_dir: str = None):
        """
        :param measure:          The target measure to be used for parameter tuning
        :param measure_is_loss:  True, if the target measure is a loss, False otherwise
        :param output_dir:       The path of the directory where the evaluation results should be stored
        """
        self.target_measure = measure
        self.target_measure_is_loss = measure_is_loss
        self.best_configuration = best_configuration
        evaluation_outputs = [EvaluationLogOutput(output_individual_folds=False)]

        if output_dir is not None:
            evaluation_outputs.append(EvaluationCsvOutput(output_dir=output_dir, output_individual_folds=False))

        self.evaluation = ClassificationEvaluation(*evaluation_outputs)
        self.indices_map = None

    def evaluate(self, configurations: List[dict], ground_truth_tuning: np.ndarray, predictions_tuning: np.ndarray,
                 ground_truth_test: np.ndarray, predictions_test: np.ndarray, current_bootstrap: int,
                 num_bootstraps: int):
        measure = self.target_measure
        measure_is_loss = self.target_measure_is_loss
        best_configuration = self.best_configuration
        evaluation = self.evaluation
        indices_map = self.indices_map

        if indices_map is None:
            indices_map = {}

            for c, config in enumerate(configurations):
                if best_configuration is None or self.__is_best_config(config, best_configuration):
                    num_rules = config['num_rules']
                    indices_list = indices_map[num_rules] if num_rules in indices_map else []
                    indices_list.append(c)
                    indices_map[num_rules] = indices_list

            self.indices_map = indices_map

        for num_rules, indices_list in indices_map.items():
            num_configurations = len(indices_list)

            if num_configurations > 1:
                evaluation_scores_tuning = np.empty(num_configurations, dtype=float)

                for k, index in enumerate(indices_list):
                    predictions = predictions_tuning[:, index, :]
                    evaluation_scores_tuning[k] = measure(ground_truth_tuning, predictions)

                best_k = np.argmin(evaluation_scores_tuning) if measure_is_loss else np.argmax(evaluation_scores_tuning)
                best_index = indices_list[best_k.item()]
            else:
                best_index = indices_list[0]

            best_predictions = predictions_test[:, best_index, :]
            evaluation_name = 'best_configuration_num-rules=' + str(num_rules)
            evaluation.evaluate(evaluation_name, best_predictions, ground_truth_test, first_fold=0,
                                current_fold=current_bootstrap, last_fold=num_bootstraps - 1, num_folds=num_bootstraps)

    @staticmethod
    def __is_best_config(config: dict, best: dict):
        for key, value in config.items():
            if key != 'num_rules' and best[key] != value:
                return False
        return True


def __create_configurations(arguments) -> List[dict]:
    num_rules_values: List[int] = arguments.num_rules
    loss_values: List[str] = arguments.loss
    head_refinement_values: List[str] = [None if x.lower() == 'none' else x for x in arguments.head_refinement]
    label_sub_sampling_values: List[int] = arguments.label_sub_sampling
    instance_sub_sampling_values: List[str] = [None if x.lower() == 'none' else x for x in
                                               arguments.instance_sub_sampling]
    feature_sub_sampling_values: List[str] = [None if x.lower() == 'none' else x for x in
                                              arguments.feature_sub_sampling]
    pruning_values: List[str] = [None if x.lower() == 'none' else x for x in arguments.pruning]
    shrinkage_values: List[float] = arguments.shrinkage
    l2_regularization_weight_values: List[float] = arguments.l2_regularization_weight
    result: List[dict] = []

    for num_rules in num_rules_values:
        for loss in loss_values:
            for pruning in pruning_values:
                for instance_sub_sampling in instance_sub_sampling_values:
                    for feature_sub_sampling in feature_sub_sampling_values:
                        for shrinkage in shrinkage_values:
                            for l2_regularization_weight in l2_regularization_weight_values:
                                for head_refinement in head_refinement_values:
                                    for label_sub_sampling in label_sub_sampling_values:
                                        if head_refinement == 'full' or label_sub_sampling == -1:
                                            configuration = {
                                                'num_rules': num_rules,
                                                'loss': loss,
                                                'pruning': pruning,
                                                'instance_sub_sampling': instance_sub_sampling,
                                                'feature_sub_sampling': feature_sub_sampling,
                                                'shrinkage': shrinkage,
                                                'l2_regularization_weight': l2_regularization_weight,
                                                'head_refinement': head_refinement,
                                                'label_sub_sampling': label_sub_sampling
                                            }
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
    parser.add_argument('--min-rules', type=int, default=-1,
                        help='The minimum number of rules to be used for testing models')
    parser.add_argument('--max-rules', type=int, default=-1,
                        help='The maximum number of rules to be used for testing models')
    parser.add_argument('--step-size-rules', type=int, default=50,
                        help='The step size to be used for testing subsets of a model\'s rules')
    parser.add_argument('--target-measure', type=target_measure, default='hamming-loss',
                        help='The target measure to be used for evaluating different configurations on the tuning set')
    parser.add_argument('--num-rules', type=int_list, default='500',
                        help='The values for the parameter \'num_rules\' as a comma-separated list')
    parser.add_argument('--loss', type=string_list, default='macro-squared-error-loss',
                        help='The values for the parameter \'loss\' as a comma-separated list')
    parser.add_argument('--head-refinement', type=string_list, default='single-label',
                        help='The values for the parameter \'head_refinement\' as a comma-separated list')
    parser.add_argument('--label-sub-sampling', type=int_list, default='-1',
                        help='The values for the parameter \'label_sub_sampling\' as a comma-separated list')
    parser.add_argument('--instance-sub-sampling', type=string_list, default='None',
                        help='The values for the parameter \'instance_sub_sampling\' as a comma-separated list')
    parser.add_argument('--feature-sub-sampling', type=string_list, default='None',
                        help='The values for the parameter \'feature_sub_sampling\' as a comma-separated list')
    parser.add_argument('--pruning', type=string_list, default='None',
                        help='The values for the parameter \'pruning\' as a comma-separated list')
    parser.add_argument('--shrinkage', type=float_list, default='1.0',
                        help='The values for the parameter \'shrinkage\' as a comma-separated list')
    parser.add_argument('--l2-regularization-weight', type=float_list, default='1.0',
                        help='The values for the parameter \'l2-regularization-weight\' as a comma-separated list')
    parser.add_argument('--subset-correction', type=boolean_string, default='False',
                        help='True if subset correction should be used, False otherwise')
    parser.add_argument('--mode', type=str, default='default',
                        help='The mode to be used. Must be \'evaluate\', \'print-best\' or \'evaluate-best\'')
    args = parser.parse_args()
    log.basicConfig(level=args.log_level)
    log.info('Configuration: %s', args)

    target_measure, target_measure_is_loss = args.target_measure
    base_configurations = __create_configurations(args)
    learner = Boomer()
    bbc_cv_adapter = BoomerBccCvAdapter(data_dir=args.data_dir, data_set=args.dataset, num_folds=args.folds,
                                        model_dir=args.model_dir, min_rules=args.min_rules, max_rules=args.max_rules,
                                        step_size_rules=args.step_size_rules, subset_correction=args.subset_correction)
    # bootstrapping = CVBootstrapping(data_dir=args.data_dir, data_set=args.dataset, num_folds=args.folds)
    bootstrapping = DefaultBootstrapping(args.num_bootstraps)
    bbc_cv = BbcCv(configurations=base_configurations, adapter=bbc_cv_adapter, bootstrapping=bootstrapping,
                   learner=learner)
    bbc_cv.random_state = args.random_state
    bbc_cv.store_predictions()
    mode = args.mode

    if mode == 'evaluate':
        bbc_cv.evaluate(observer=DefaultBbcCvObserver(output_dir=args.output_dir, target_measure=target_measure,
                                                      target_measure_is_loss=target_measure_is_loss))
    elif mode == 'print-best':
        tuning_evaluation_observer = TuningEvaluationBbcCvObserver(measure=target_measure)
        bbc_cv.evaluate(observer=tuning_evaluation_observer)
        tuning_scores = tuning_evaluation_observer.evaluation_scores_tuning

        if not target_measure_is_loss:
            tuning_scores = 1 - tuning_scores

        ranks = np.empty_like(tuning_scores)

        for i in range(tuning_scores.shape[1]):
            ranks[:, i] = stats.rankdata(tuning_scores[:, i], method='average')

        avg_ranks = np.average(ranks, axis=1)
        base_configurations = bbc_cv.configurations_
        best_config = base_configurations[np.argmin(avg_ranks).item()]
        log.info('Best configuration: %s', str(best_config))
    elif mode == 'evaluate-best':
        best_config = None
        best_config_observer = BestConfigBbcCvObserver(measure=target_measure,
                                                       measure_is_loss=target_measure_is_loss,
                                                       best_configuration=best_config, output_dir=args.output_dir)
        bbc_cv.evaluate(observer=best_config_observer)
    else:
        raise ValueError('Invalid value given for argument \'mode\': ' + str(mode))
