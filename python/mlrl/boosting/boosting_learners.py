#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides a scikit-learn implementations of boosting algorithms
"""
import logging as log
from typing import Optional

from mlrl.boosting.cython.losses_example_wise import ExampleWiseLogisticLoss
from mlrl.boosting.cython.losses_label_wise import LabelWiseLoss, LabelWiseLogisticLoss, LabelWiseSquaredErrorLoss, \
    LabelWiseSquaredHingeLoss
from mlrl.boosting.cython.model import RuleListBuilder
from mlrl.boosting.cython.output import LabelWiseClassificationPredictor, ExampleWiseClassificationPredictor, \
    LabelWiseProbabilityPredictor, LabelWiseTransformationFunction, LogisticFunction
from mlrl.boosting.cython.post_processing import ConstantShrinkage
from mlrl.boosting.cython.rule_evaluation_example_wise import RegularizedExampleWiseRuleEvaluationFactory, \
    EqualWidthBinningExampleWiseRuleEvaluationFactory
from mlrl.boosting.cython.rule_evaluation_label_wise import RegularizedLabelWiseRuleEvaluationFactory, \
    EqualWidthBinningLabelWiseRuleEvaluationFactory
from mlrl.boosting.cython.statistics_example_wise import ExampleWiseStatisticsProviderFactory
from mlrl.boosting.cython.statistics_label_wise import LabelWiseStatisticsProviderFactory
from mlrl.common.cython.head_refinement import HeadRefinementFactory, SingleLabelHeadRefinementFactory, \
    FullHeadRefinementFactory
from mlrl.common.cython.input import CContiguousLabelMatrix
from mlrl.common.cython.model import ModelBuilder
from mlrl.common.cython.output import Predictor
from mlrl.common.cython.post_processing import PostProcessor, NoPostProcessor
from mlrl.common.cython.rule_induction import TopDownRuleInduction, SequentialRuleModelInduction
from mlrl.common.cython.statistics import StatisticsProviderFactory
from mlrl.common.cython.stopping import MeasureStoppingCriterion, AggregationFunction, MinFunction, MaxFunction, \
    ArithmeticMeanFunction
from sklearn.base import ClassifierMixin

from mlrl.common.rule_learners import INSTANCE_SUB_SAMPLING_BAGGING, FEATURE_SUB_SAMPLING_RANDOM, \
    HEAD_REFINEMENT_SINGLE, ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS
from mlrl.common.rule_learners import MLRuleLearner, SparsePolicy
from mlrl.common.rule_learners import create_pruning, create_feature_sub_sampling, create_instance_sub_sampling, \
    create_label_sub_sampling, create_partition_sampling, create_max_conditions, create_stopping_criteria, \
    create_min_coverage, create_max_head_refinements, get_preferred_num_threads, create_thresholds_factory, \
    parse_prefix_and_dict, get_int_argument, get_float_argument, get_string_argument, get_bool_argument

EARLY_STOPPING_MEASURE = 'measure'

AGGREGATION_FUNCTION_MIN = 'min'

AGGREGATION_FUNCTION_MAX = 'max'

AGGREGATION_FUNCTION_ARITHMETIC_MEAN = 'avg'

ARGUMENT_MIN_RULES = 'min_rules'

ARGUMENT_UPDATE_INTERVAL = 'update_interval'

ARGUMENT_STOP_INTERVAL = 'stop_interval'

ARGUMENT_NUM_PAST = 'num_past'

ARGUMENT_NUM_RECENT = 'num_recent'

ARGUMENT_MIN_IMPROVEMENT = 'min_improvement'

ARGUMENT_FORCE_STOP = 'force_stop'

ARGUMENT_AGGREGATION_FUNCTION = 'aggregation'

HEAD_REFINEMENT_FULL = 'full'

LOSS_LABEL_WISE_LOGISTIC = 'label-wise-logistic-loss'

LOSS_LABEL_WISE_SQUARED_ERROR = 'label-wise-squared-error-loss'

LOSS_LABEL_WISE_SQUARED_HINGE = 'label-wise-squared-hinge-loss'

LOSS_EXAMPLE_WISE_LOGISTIC = 'example-wise-logistic-loss'

NON_DECOMPOSABLE_LOSSES = {LOSS_EXAMPLE_WISE_LOGISTIC}

PREDICTOR_LABEL_WISE = 'label-wise'

PREDICTOR_EXAMPLE_WISE = 'example-wise'

LABEL_BINNING_EQUAL_WIDTH = 'equal-width'


class Boomer(MLRuleLearner, ClassifierMixin):
    """
    A scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
    classification rules.
    """

    def __init__(self, random_state: int = 1, feature_format: str = SparsePolicy.AUTO.value,
                 label_format: str = SparsePolicy.AUTO.value, max_rules: int = 1000, time_limit: int = -1,
                 early_stopping: str = None, head_refinement: str = None, loss: str = LOSS_LABEL_WISE_LOGISTIC,
                 predictor: str = None, label_sub_sampling: str = None,
                 instance_sub_sampling: str = INSTANCE_SUB_SAMPLING_BAGGING,
                 feature_sub_sampling: str = FEATURE_SUB_SAMPLING_RANDOM, holdout_set_size: float = 0.0,
                 feature_binning: str = None, label_binning: str = None, pruning: str = None, shrinkage: float = 0.3,
                 l2_regularization_weight: float = 1.0, min_coverage: int = 1, max_conditions: int = -1,
                 max_head_refinements: int = 1, num_threads_refinement: int = 1, num_threads_update: int = 1,
                 num_threads_prediction: int = 1):
        """
        :param max_rules:                           The maximum number of rules to be induced (including the default
                                                    rule)
        :param time_limit:                          The duration in seconds after which the induction of rules should be
                                                    canceled
        :param early_stopping:                      The strategy that is used for early stopping. Must be `measure` or
                                                    None, if no early stopping should be used
        :param head_refinement:                     The strategy that is used to find the heads of rules. Must be
                                                    `single-label`, `full` or None, if the default strategy should be
                                                    used
        :param loss:                                The loss function to be minimized. Must be
                                                    `label-wise-squared-error-loss`, `label-wise-logistic-loss` or
                                                    `example-wise-logistic-loss`
        :param predictor:                           The strategy that is used for making predictions. Must be
                                                    `label-wise`, `example-wise` or None, if the default strategy should
                                                    be used
        :param label_sub_sampling:                  The strategy that is used for sub-sampling the labels each time a
                                                    new classification rule is learned. Must be 'random-label-selection'
                                                    or None, if no sub-sampling should be used. Additional arguments may
                                                    be provided as a dictionary, e.g.
                                                    `random-label-selection{\"num_samples\":5}`
        :param instance_sub_sampling:               The strategy that is used for sub-sampling the training examples
                                                    each time a new classification rule is learned. Must be `bagging`,
                                                    `random-instance-selection` or None, if no sub-sampling should be
                                                    used. Additional arguments may be provided as a dictionary, e.g.
                                                    `bagging{\"sample_size\":0.5}`
        :param feature_sub_sampling:                The strategy that is used for sub-sampling the features each time a
                                                    classification rule is refined. Must be `random-feature-selection`
                                                    or None, if no sub-sampling should be used. Additional argument may
                                                    be provided as a dictionary, e.g.
                                                    `random-feature-selection{\"sample_size\":0.5}`
        :param holdout_set_size:                    The fraction of the training examples that should be included in the
                                                    holdout set. Must be in (0, 1) or 0, if no holdout set should be
                                                    used
        :param feature_binning:                     The strategy that is used for assigning examples to bins based on
                                                    their feature values. Must be `equal-width`, `equal-frequency` or
                                                    None, if no feature binning should be used. Additional arguments may
                                                    be provided as a dictionary, e.g. `equal-width{\"bin_ratio\":0.5}`
        :param label_binning:                       The strategy that is used for assigning labels to bins. Must be
                                                    `equal-width` or None, if no label binning should be used.
                                                    Additional arguments may be provided as a dictionary, e.g.
                                                    `equal-width{\"num_positive_bins\":8, \"num_negative_bins\":8}`
        :param pruning:                             The strategy that is used for pruning rules. Must be `irep` or None,
                                                    if no pruning should be used
        :param shrinkage:                           The shrinkage parameter that should be applied to the predictions of
                                                    newly induced rules to reduce their effect on the entire model. Must
                                                    be in (0, 1]
        :param l2_regularization_weight:            The weight of the L2 regularization that is applied for calculating
                                                    the scores that are predicted by rules. Must be at least 0
        :param min_coverage:                        The minimum number of training examples that must be covered by a
                                                    rule. Must be at least 1
        :param max_conditions:                      The maximum number of conditions to be included in a rule's body.
                                                    Must be at least 1 or -1, if the number of conditions should not be
                                                    restricted
        :param max_head_refinements:                The maximum number of times the head of a rule may be refined after
                                                    a new condition has been added to its body. Must be at least 1 or
                                                    -1, if the number of refinements should not be restricted
        :param num_threads_refinement:              The number of threads to be used to search for potential refinements
                                                    of rules or -1, if the number of cores that are available on the
                                                    machine should be used
        :param num_threads_update:                  The number of threads to be used to update statistics or -1, if the
                                                    number of cores that are available on the machine should be used
        :param num_threads_prediction:              The number of threads to be used to make predictions or -1, if the
                                                    number of cores that are available on the machine should be used
        """
        super().__init__(random_state, feature_format, label_format)
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.early_stopping = early_stopping
        self.head_refinement = head_refinement
        self.loss = loss
        self.predictor = predictor
        self.label_sub_sampling = label_sub_sampling
        self.instance_sub_sampling = instance_sub_sampling
        self.feature_sub_sampling = feature_sub_sampling
        self.holdout_set_size = holdout_set_size
        self.feature_binning = feature_binning
        self.label_binning = label_binning
        self.pruning = pruning
        self.shrinkage = shrinkage
        self.l2_regularization_weight = l2_regularization_weight
        self.min_coverage = min_coverage
        self.max_conditions = max_conditions
        self.max_head_refinements = max_head_refinements
        self.num_threads_refinement = num_threads_refinement
        self.num_threads_update = num_threads_update
        self.num_threads_prediction = num_threads_prediction

    def get_name(self) -> str:
        name = 'max-rules=' + str(self.max_rules)
        if self.early_stopping is not None:
            name += '_early-stopping=' + str(self.early_stopping)
        if self.head_refinement is not None:
            name += '_head-refinement=' + str(self.head_refinement)
        name += '_loss=' + str(self.loss)
        if self.predictor is not None:
            name += '_predictor=' + str(self.predictor)
        if self.label_sub_sampling is not None:
            name += '_label-sub-sampling=' + str(self.label_sub_sampling)
        if self.instance_sub_sampling is not None:
            name += '_instance-sub-sampling=' + str(self.instance_sub_sampling)
        if self.feature_sub_sampling is not None:
            name += '_feature-sub-sampling=' + str(self.feature_sub_sampling)
        if float(self.holdout_set_size) > 0.0:
            name += '_holdout=' + str(self.holdout_set_size)
        if self.feature_binning is not None:
            name += '_feature-binning=' + str(self.feature_binning)
        if self.label_binning is not None:
            name += '_label-binning=' + str(self.label_binning)
        if self.pruning is not None:
            name += '_pruning=' + str(self.pruning)
        if 0.0 < float(self.shrinkage) < 1.0:
            name += '_shrinkage=' + str(self.shrinkage)
        if float(self.l2_regularization_weight) > 0.0:
            name += '_l2=' + str(self.l2_regularization_weight)
        if int(self.min_coverage) > 1:
            name += '_min-coverage=' + str(self.min_coverage)
        if int(self.max_conditions) != -1:
            name += '_max-conditions=' + str(self.max_conditions)
        if int(self.max_head_refinements) != -1:
            name += '_max-head-refinements=' + str(self.max_head_refinements)
        if int(self.random_state) != 1:
            name += '_random_state=' + str(self.random_state)
        return name

    def _create_predictor(self, num_labels: int, label_matrix: CContiguousLabelMatrix) -> Predictor:
        predictor = self.__get_preferred_predictor()

        if predictor == PREDICTOR_LABEL_WISE:
            return self.__create_label_wise_predictor(num_labels)
        elif predictor == PREDICTOR_EXAMPLE_WISE:
            return self.__create_example_wise_predictor(label_matrix)
        raise ValueError('Invalid value given for parameter \'predictor\': ' + str(predictor))

    def _create_predictor_lil(self, num_labels: int, label_matrix: list) -> Predictor:
        predictor = self.__get_preferred_predictor()

        if predictor == PREDICTOR_LABEL_WISE:
            return self.__create_label_wise_predictor(num_labels)
        elif predictor == PREDICTOR_EXAMPLE_WISE:
            return self.__create_example_wise_predictor_lil(num_labels, label_matrix)
        raise ValueError('Invalid value given for parameter \'predictor\': ' + str(predictor))

    def _create_probability_predictor(self, num_labels: int, label_matrix: CContiguousLabelMatrix) -> Predictor:
        predictor = self.__get_preferred_predictor()

        if predictor == PREDICTOR_LABEL_WISE and self.loss == LOSS_LABEL_WISE_LOGISTIC:
            transformation_function = LogisticFunction()
            return self.__create_label_wise_probability_predictor(num_labels, transformation_function)

        return None

    def _create_probability_predictor_lil(self, num_labels: int, label_matrix: list) -> Predictor:
        predictor = self.__get_preferred_predictor()

        if predictor == PREDICTOR_LABEL_WISE and self.loss == LOSS_LABEL_WISE_LOGISTIC:
            transformation_function = LogisticFunction()
            return self.__create_label_wise_probability_predictor(num_labels, transformation_function)

        return None

    def __get_preferred_predictor(self) -> str:
        predictor = self.predictor

        if predictor is None:
            if self.loss in NON_DECOMPOSABLE_LOSSES:
                return PREDICTOR_EXAMPLE_WISE
            else:
                return PREDICTOR_LABEL_WISE
        return predictor

    def __create_label_wise_predictor(self, num_labels: int) -> LabelWiseClassificationPredictor:
        num_threads = get_preferred_num_threads(self.num_threads_prediction)
        threshold = 0.5 if self.loss == LOSS_LABEL_WISE_SQUARED_HINGE else 0.0
        return LabelWiseClassificationPredictor(num_labels=num_labels, threshold=threshold, num_threads=num_threads)

    def __create_label_wise_probability_predictor(
            self, num_labels: int,
            transformation_function: LabelWiseTransformationFunction) -> LabelWiseProbabilityPredictor:
        num_threads = get_preferred_num_threads(self.num_threads_prediction)
        return LabelWiseProbabilityPredictor(num_labels=num_labels, transformation_function=transformation_function,
                                             num_threads=num_threads)

    def __create_example_wise_predictor(self,
                                        label_matrix: CContiguousLabelMatrix) -> ExampleWiseClassificationPredictor:
        loss = self.__create_loss_function()
        num_threads = get_preferred_num_threads(self.num_threads_prediction)
        return ExampleWiseClassificationPredictor.create(label_matrix, loss, num_threads)

    def __create_example_wise_predictor_lil(self, num_labels: int,
                                            label_matrix: list) -> ExampleWiseClassificationPredictor:
        loss = self.__create_loss_function()
        num_threads = get_preferred_num_threads(self.num_threads_prediction)
        return ExampleWiseClassificationPredictor.create_lil(num_labels, label_matrix, loss, num_threads)

    def _create_model_builder(self) -> ModelBuilder:
        return RuleListBuilder()

    def _create_rule_model_induction(self, num_labels: int) -> SequentialRuleModelInduction:
        stopping_criteria = create_stopping_criteria(int(self.max_rules), int(self.time_limit))
        early_stopping_criterion = self.__create_early_stopping()
        if early_stopping_criterion is not None:
            stopping_criteria.append(early_stopping_criterion)
        label_sub_sampling = create_label_sub_sampling(self.label_sub_sampling, num_labels)
        instance_sub_sampling = create_instance_sub_sampling(self.instance_sub_sampling)
        feature_sub_sampling = create_feature_sub_sampling(self.feature_sub_sampling)
        partition_sampling = create_partition_sampling(self.holdout_set_size)
        pruning = create_pruning(self.pruning, self.instance_sub_sampling)
        shrinkage = self.__create_post_processor()
        min_coverage = create_min_coverage(self.min_coverage)
        max_conditions = create_max_conditions(self.max_conditions)
        max_head_refinements = create_max_head_refinements(self.max_head_refinements)
        loss_function = self.__create_loss_function()
        default_rule_head_refinement_factory = FullHeadRefinementFactory()
        head_refinement_factory = self.__create_head_refinement_factory()
        l2_regularization_weight = self.__create_l2_regularization_weight()
        rule_evaluation_factory = self.__create_rule_evaluation_factory(loss_function, l2_regularization_weight)
        num_threads_update = get_preferred_num_threads(self.num_threads_update)
        statistics_provider_factory = self.__create_statistics_provider_factory(loss_function, rule_evaluation_factory,
                                                                                num_threads_update)
        thresholds_factory = create_thresholds_factory(self.feature_binning, num_threads_update)
        num_threads_refinement = get_preferred_num_threads(self.num_threads_refinement)
        rule_induction = TopDownRuleInduction(num_threads_refinement)
        return SequentialRuleModelInduction(statistics_provider_factory, thresholds_factory, rule_induction,
                                            default_rule_head_refinement_factory, head_refinement_factory,
                                            label_sub_sampling, instance_sub_sampling, feature_sub_sampling,
                                            partition_sampling, pruning, shrinkage, min_coverage, max_conditions,
                                            max_head_refinements, stopping_criteria)

    def __create_early_stopping(self) -> Optional[MeasureStoppingCriterion]:
        early_stopping = self.early_stopping

        if early_stopping is None:
            return None
        else:
            prefix, args = parse_prefix_and_dict(early_stopping, [EARLY_STOPPING_MEASURE])

            if prefix == EARLY_STOPPING_MEASURE:
                if self.holdout_set_size <= 0.0:
                    log.warning('Parameter \'early_stopping\' does not have any effect, because parameter \'holdout\' '
                                + 'is set to \'0\'!')
                    return None
                else:
                    loss = self.__create_loss_function()
                    aggregation_function = self.__create_aggregation_function(
                        get_string_argument(args, ARGUMENT_AGGREGATION_FUNCTION, 'avg'))
                    min_rules = get_int_argument(args, ARGUMENT_MIN_RULES, 100, lambda x: 1 <= x)
                    update_interval = get_int_argument(args, ARGUMENT_UPDATE_INTERVAL, 1, lambda x: 1 <= x)
                    stop_interval = get_int_argument(args, ARGUMENT_STOP_INTERVAL, 1,
                                                     lambda x: 1 <= x and x % update_interval == 0)
                    num_past = get_int_argument(args, ARGUMENT_NUM_PAST, 50, lambda x: 1 <= x)
                    num_recent = get_int_argument(args, ARGUMENT_NUM_RECENT, 50, lambda x: 1 <= x)
                    min_improvement = get_float_argument(args, ARGUMENT_MIN_IMPROVEMENT, 0.005, lambda x: 0 <= x <= 1)
                    force_stop = get_bool_argument(args, ARGUMENT_FORCE_STOP, True)
                    return MeasureStoppingCriterion(loss, aggregation_function, min_rules=min_rules,
                                                    update_interval=update_interval, stop_interval=stop_interval,
                                                    num_past=num_past, num_recent=num_recent,
                                                    min_improvement=min_improvement, force_stop=force_stop)
            raise ValueError('Invalid value given for parameter \'early_stopping\': ' + str(early_stopping))

    def __create_aggregation_function(self, aggregation_function: str) -> AggregationFunction:
        if aggregation_function == AGGREGATION_FUNCTION_MIN:
            return MinFunction()
        elif aggregation_function == AGGREGATION_FUNCTION_MAX:
            return MaxFunction()
        elif aggregation_function == AGGREGATION_FUNCTION_ARITHMETIC_MEAN:
            return ArithmeticMeanFunction()
        raise ValueError('Invalid value given for argument \'' + ARGUMENT_AGGREGATION_FUNCTION + '\': '
                         + str(aggregation_function))

    def __create_l2_regularization_weight(self) -> float:
        l2_regularization_weight = float(self.l2_regularization_weight)

        if l2_regularization_weight < 0:
            raise ValueError(
                'Invalid value given for parameter \'l2_regularization_weight\': ' + str(l2_regularization_weight))

        return l2_regularization_weight

    def __create_loss_function(self):
        loss = self.loss

        if loss == LOSS_LABEL_WISE_SQUARED_ERROR:
            return LabelWiseSquaredErrorLoss()
        elif loss == LOSS_LABEL_WISE_SQUARED_HINGE:
            return LabelWiseSquaredHingeLoss()
        elif loss == LOSS_LABEL_WISE_LOGISTIC:
            return LabelWiseLogisticLoss()
        elif loss == LOSS_EXAMPLE_WISE_LOGISTIC:
            return ExampleWiseLogisticLoss()
        raise ValueError('Invalid value given for parameter \'loss\': ' + str(loss))

    def __create_rule_evaluation_factory(self, loss_function, l2_regularization_weight: float):
        label_binning, bin_ratio, min_bins, max_bins = self.__create_label_binning()

        if isinstance(loss_function, LabelWiseLoss):
            if label_binning == LABEL_BINNING_EQUAL_WIDTH:
                return EqualWidthBinningLabelWiseRuleEvaluationFactory(l2_regularization_weight, bin_ratio, min_bins,
                                                                       max_bins)
            else:
                return RegularizedLabelWiseRuleEvaluationFactory(l2_regularization_weight)
        else:
            if label_binning == LABEL_BINNING_EQUAL_WIDTH:
                return EqualWidthBinningExampleWiseRuleEvaluationFactory(l2_regularization_weight, bin_ratio, min_bins,
                                                                         max_bins)
            else:
                return RegularizedExampleWiseRuleEvaluationFactory(l2_regularization_weight)

    def __create_label_binning(self) -> (str, float):
        label_binning = self.label_binning

        if label_binning is None:
            return None, 0, 0, 0
        else:
            prefix, args = parse_prefix_and_dict(label_binning, [LABEL_BINNING_EQUAL_WIDTH])

            if prefix == LABEL_BINNING_EQUAL_WIDTH:
                bin_ratio = get_float_argument(args, ARGUMENT_BIN_RATIO, 0.04, lambda x: 0 < x < 1)
                min_bins = get_int_argument(args, ARGUMENT_MIN_BINS, 1, lambda x: x >= 1)
                max_bins = get_int_argument(args, ARGUMENT_MAX_BINS, 0, lambda x: x == 0 or x >= min_bins)
                return prefix, bin_ratio, min_bins, max_bins
            raise ValueError('Invalid value given for parameter \'label_binning\': ' + str(label_binning))

    def __create_statistics_provider_factory(self, loss_function, rule_evaluation_factory,
                                             num_threads: int) -> StatisticsProviderFactory:
        if isinstance(loss_function, LabelWiseLoss):
            return LabelWiseStatisticsProviderFactory(loss_function, rule_evaluation_factory, rule_evaluation_factory,
                                                      num_threads)
        else:
            return ExampleWiseStatisticsProviderFactory(loss_function, rule_evaluation_factory, rule_evaluation_factory,
                                                        num_threads)

    def __create_head_refinement_factory(self) -> HeadRefinementFactory:
        head_refinement = self.___get_preferred_head_refinement()

        if head_refinement == HEAD_REFINEMENT_SINGLE:
            return SingleLabelHeadRefinementFactory()
        elif head_refinement == HEAD_REFINEMENT_FULL:
            return FullHeadRefinementFactory()
        raise ValueError('Invalid value given for parameter \'head_refinement\': ' + str(head_refinement))

    def ___get_preferred_head_refinement(self) -> str:
        head_refinement = self.head_refinement

        if head_refinement is None:
            if self.loss in NON_DECOMPOSABLE_LOSSES:
                return HEAD_REFINEMENT_FULL
            else:
                return HEAD_REFINEMENT_SINGLE
        return head_refinement

    def __create_post_processor(self) -> PostProcessor:
        shrinkage = float(self.shrinkage)

        if 0.0 < shrinkage < 1.0:
            return ConstantShrinkage(shrinkage)
        if shrinkage == 1.0:
            return NoPostProcessor()
        raise ValueError('Invalid value given for parameter \'shrinkage\': ' + str(shrinkage))
