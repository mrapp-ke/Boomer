#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides scikit-learn implementations of boosting algorithms.
"""
import logging as log
from typing import Optional, Dict, Set, List

from mlrl.boosting.cython.label_binning import LabelBinningFactory, EqualWidthLabelBinningFactory
from mlrl.boosting.cython.losses_example_wise import ExampleWiseLogisticLoss
from mlrl.boosting.cython.losses_label_wise import LabelWiseLoss, LabelWiseLogisticLoss, LabelWiseSquaredErrorLoss, \
    LabelWiseSquaredHingeLoss
from mlrl.boosting.cython.model import RuleListBuilder
from mlrl.boosting.cython.output import LabelWiseClassificationPredictor, ExampleWiseClassificationPredictor, \
    LabelWiseProbabilityPredictor, LabelWiseTransformationFunction, LogisticFunction
from mlrl.boosting.cython.post_processing import ConstantShrinkage
from mlrl.boosting.cython.rule_evaluation_example_wise import ExampleWiseCompleteRuleEvaluationFactory, \
    ExampleWiseCompleteBinnedRuleEvaluationFactory
from mlrl.boosting.cython.rule_evaluation_label_wise import LabelWiseSingleLabelRuleEvaluationFactory, \
    LabelWiseCompleteRuleEvaluationFactory, LabelWiseCompleteBinnedRuleEvaluationFactory
from mlrl.boosting.cython.statistics_example_wise import DenseExampleWiseStatisticsProviderFactory, \
    DenseConvertibleExampleWiseStatisticsProviderFactory
from mlrl.boosting.cython.statistics_label_wise import DenseLabelWiseStatisticsProviderFactory
from mlrl.common.cython.feature_sampling import FeatureSamplingFactory
from mlrl.common.cython.input import LabelMatrix, LabelVectorSet
from mlrl.common.cython.instance_sampling import InstanceSamplingFactory
from mlrl.common.cython.label_sampling import LabelSamplingFactory
from mlrl.common.cython.model import ModelBuilder
from mlrl.common.cython.output import Predictor
from mlrl.common.cython.partition_sampling import PartitionSamplingFactory
from mlrl.common.cython.post_processing import PostProcessor, NoPostProcessor
from mlrl.common.cython.pruning import Pruning
from mlrl.common.cython.rule_induction import RuleInduction, TopDownRuleInduction
from mlrl.common.cython.rule_model_assemblage import RuleModelAssemblageFactory, SequentialRuleModelAssemblageFactory
from mlrl.common.cython.statistics import StatisticsProviderFactory
from mlrl.common.cython.stopping import StoppingCriterion, MeasureStoppingCriterion, AggregationFunction, MinFunction, \
    MaxFunction, ArithmeticMeanFunction
from mlrl.common.cython.thresholds import ThresholdsFactory
from mlrl.common.options import BooleanOption
from mlrl.common.rule_learners import AUTOMATIC, SAMPLING_WITHOUT_REPLACEMENT, HEAD_TYPE_SINGLE, ARGUMENT_BIN_RATIO, \
    ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS, ARGUMENT_NUM_THREADS
from mlrl.common.rule_learners import MLRuleLearner, SparsePolicy, FeatureCharacteristics, LabelCharacteristics
from mlrl.common.rule_learners import create_pruning, create_feature_sampling_factory, create_label_sampling_factory, \
    create_instance_sampling_factory, create_partition_sampling_factory, create_stopping_criteria, \
    create_num_threads, create_thresholds_factory, parse_param, parse_param_and_options
from sklearn.base import ClassifierMixin

EARLY_STOPPING_LOSS = 'loss'

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

HEAD_TYPE_COMPLETE = 'complete'

LOSS_LOGISTIC_LABEL_WISE = 'logistic-label-wise'

LOSS_LOGISTIC_EXAMPLE_WISE = 'logistic-example-wise'

LOSS_SQUARED_ERROR_LABEL_WISE = 'squared-error-label-wise'

LOSS_SQUARED_HINGE_LABEL_WISE = 'squared-hinge-label-wise'

NON_DECOMPOSABLE_LOSSES = {LOSS_LOGISTIC_EXAMPLE_WISE}

PREDICTOR_LABEL_WISE = 'label-wise'

PREDICTOR_EXAMPLE_WISE = 'example-wise'

LABEL_BINNING_EQUAL_WIDTH = 'equal-width'

HEAD_TYPE_VALUES: Set[str] = {HEAD_TYPE_SINGLE, HEAD_TYPE_COMPLETE, AUTOMATIC}

EARLY_STOPPING_VALUES: Dict[str, Set[str]] = {
    EARLY_STOPPING_LOSS: {ARGUMENT_AGGREGATION_FUNCTION, ARGUMENT_MIN_RULES, ARGUMENT_UPDATE_INTERVAL,
                          ARGUMENT_STOP_INTERVAL, ARGUMENT_NUM_PAST, ARGUMENT_NUM_RECENT, ARGUMENT_MIN_IMPROVEMENT,
                          ARGUMENT_FORCE_STOP}
}

LABEL_BINNING_VALUES: Dict[str, Set[str]] = {
    LABEL_BINNING_EQUAL_WIDTH: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS},
    AUTOMATIC: {}
}

LOSS_VALUES: Set[str] = {LOSS_SQUARED_ERROR_LABEL_WISE, LOSS_SQUARED_HINGE_LABEL_WISE, LOSS_LOGISTIC_LABEL_WISE,
                         LOSS_LOGISTIC_EXAMPLE_WISE}

PREDICTOR_VALUES: Set[str] = {PREDICTOR_LABEL_WISE, PREDICTOR_EXAMPLE_WISE, AUTOMATIC}

PARALLEL_VALUES: Dict[str, Set[str]] = {
    str(BooleanOption.TRUE.value): {ARGUMENT_NUM_THREADS},
    str(BooleanOption.FALSE.value): {},
    AUTOMATIC: {}
}


class Boomer(MLRuleLearner, ClassifierMixin):
    """
    A scikit-multilearn implementation of "BOOMER", an algorithm for learning gradient boosted multi-label
    classification rules.
    """

    def __init__(self, random_state: int = 1, feature_format: str = SparsePolicy.AUTO.value,
                 label_format: str = SparsePolicy.AUTO.value, prediction_format: str = SparsePolicy.AUTO.value,
                 max_rules: int = 1000, default_rule: str = BooleanOption.TRUE.value, time_limit: int = 0,
                 early_stopping: str = None, head_type: str = AUTOMATIC, loss: str = LOSS_LOGISTIC_LABEL_WISE,
                 predictor: str = AUTOMATIC, label_sampling: str = None, instance_sampling: str = None,
                 recalculate_predictions: str = BooleanOption.TRUE.value,
                 feature_sampling: str = SAMPLING_WITHOUT_REPLACEMENT, holdout: str = None, feature_binning: str = None,
                 label_binning: str = AUTOMATIC, pruning: str = None, shrinkage: float = 0.3,
                 l1_regularization_weight: float = 0.0, l2_regularization_weight: float = 1.0, min_coverage: int = 1,
                 max_conditions: int = 0, max_head_refinements: int = 1, parallel_rule_refinement: str = AUTOMATIC,
                 parallel_statistic_update: str = AUTOMATIC, parallel_prediction: str = BooleanOption.TRUE.value):
        """
        :param max_rules:                           The maximum number of rules to be induced (including the default
                                                    rule)
        :param default_rule:                        Whether a default rule should be used, or not. Must be `true` or
                                                    `false`
        :param time_limit:                          The duration in seconds after which the induction of rules should be
                                                    canceled
        :param early_stopping:                      The strategy that is used for early stopping. Must be `measure` or
                                                    None, if no early stopping should be used
        :param head_type:                           The type of the rule heads that should be used. Must be
                                                    `single-label`, `complete` or 'auto', if the type of the heads
                                                    should be chosen automatically
        :param loss:                                The loss function to be minimized. Must be
                                                    `squared-error-label-wise`, `logistic-label-wise` or
                                                    `logistic-example-wise`
        :param predictor:                           The strategy that is used for making predictions. Must be
                                                    `label-wise`, `example-wise` or `auto`, if the most suitable
                                                    strategy should be chosen automatically depending on the loss
                                                    function
        :param label_sampling:                      The strategy that is used for sampling the labels each time a new
                                                    classification rule is learned. Must be 'without-replacement' or
                                                    None, if no sampling should be used. Additional options may be
                                                    provided using the bracket notation
                                                    `without-replacement{num_samples=5}`
        :param instance_sampling:                   The strategy that is used for sampling the training examples each
                                                    time a new classification rule is learned. Must be
                                                    `with-replacement`, `without-replacement` or None, if no sampling
                                                    should be used. Additional options may be provided using the bracket
                                                    notation `with-replacement{sample_size=0.5}`
        :param recalculate_predictions:             Whether the predictions of rules should be recalculated on the
                                                    entire training data, if instance sampling is used, or not. Must be
                                                    `true` or `false`
        :param feature_sampling:                    The strategy that is used for sampling the features each time a
                                                    classification rule is refined. Must be `without-replacement` or
                                                    None, if no sampling should be used. Additional options may be
                                                    provided using the bracket notation
                                                    `without-replacement{sample_size=0.5}`
        :param holdout:                             The name of the strategy to be used for creating a holdout set. Must
                                                    be `random` or None, if no holdout set should be used. Additional
                                                    options may be provided using the bracket notation
                                                    `random{holdout_set_size=0.5}`
        :param feature_binning:                     The strategy that is used for assigning examples to bins based on
                                                    their feature values. Must be `equal-width`, `equal-frequency` or
                                                    None, if no feature binning should be used. Additional options may
                                                    be provided using the bracket notation `equal-width{bin_ratio=0.5}`
        :param label_binning:                       The strategy that is used for assigning labels to bins. Must be
                                                    `auto`, `equal-width` or None, if no label binning should be used.
                                                    Additional options may be provided using the bracket notation
                                                    `equal-width{bin_ratio=0.04,min_bins=1,max_bins=0}`. If `auto` is
                                                    used, the most suitable strategy is chosen automatically based on
                                                    the loss function and the type of rule heads
        :param pruning:                             The strategy that is used for pruning rules. Must be `irep` or None,
                                                    if no pruning should be used
        :param shrinkage:                           The shrinkage parameter that should be applied to the predictions of
                                                    newly induced rules to reduce their effect on the entire model. Must
                                                    be in (0, 1]
        :param l1_regularization_weight:            The weight of the L1 regularization that is applied for calculating
                                                    the scores that are predicted by rules. Must be at least 0
        :param l2_regularization_weight:            The weight of the L2 regularization that is applied for calculating
                                                    the scores that are predicted by rules. Must be at least 0
        :param min_coverage:                        The minimum number of training examples that must be covered by a
                                                    rule. Must be at least 1
        :param max_conditions:                      The maximum number of conditions to be included in a rule's body.
                                                    Must be at least 1 or 0, if the number of conditions should not be
                                                    restricted
        :param max_head_refinements:                The maximum number of times the head of a rule may be refined after
                                                    a new condition has been added to its body. Must be at least 1 or
                                                    0, if the number of refinements should not be restricted
        :param parallel_rule_refinement:            Whether potential refinements of rules should be searched for in
                                                    parallel or not. Must be `true`, `false` or `auto`, if the most
                                                    suitable strategy should be chosen automatically depending on the
                                                    loss function. Additional options may be provided using the bracket
                                                    notation `true{num_threads=8}`
        :param parallel_statistic_update:           Whether the gradients and Hessians for different examples should be
                                                    calculated in parallel or not. Must be `true`, `false` or `auto`, if
                                                    the most suitable strategy should be chosen automatically depending
                                                    on the loss function. Additional options may be provided using the
                                                    bracket notation `true{num_threads=8}`
        :param parallel_prediction:                 Whether predictions for different examples should be obtained in
                                                    parallel or not. Must be `true` or `false`. Additional options may
                                                    be provided using the bracket notation `true{num_threads=8}`
        """
        super().__init__(random_state, feature_format, label_format, prediction_format)
        self.max_rules = max_rules
        self.default_rule = default_rule
        self.time_limit = time_limit
        self.early_stopping = early_stopping
        self.head_type = head_type
        self.loss = loss
        self.predictor = predictor
        self.label_sampling = label_sampling
        self.instance_sampling = instance_sampling
        self.recalculate_predictions = recalculate_predictions
        self.feature_sampling = feature_sampling
        self.holdout = holdout
        self.feature_binning = feature_binning
        self.label_binning = label_binning
        self.pruning = pruning
        self.shrinkage = shrinkage
        self.l1_regularization_weight = l1_regularization_weight
        self.l2_regularization_weight = l2_regularization_weight
        self.min_coverage = min_coverage
        self.max_conditions = max_conditions
        self.max_head_refinements = max_head_refinements
        self.parallel_rule_refinement = parallel_rule_refinement
        self.parallel_statistic_update = parallel_statistic_update
        self.parallel_prediction = parallel_prediction

    def get_name(self) -> str:
        name = 'max-rules=' + str(self.max_rules)
        if self.early_stopping is not None:
            name += '_early-stopping=' + str(self.early_stopping)
        if self.head_type != AUTOMATIC:
            name += '_head-type=' + str(self.head_type)
        name += '_loss=' + str(self.loss)
        if self.predictor != AUTOMATIC:
            name += '_predictor=' + str(self.predictor)
        if self.label_sampling is not None:
            name += '_label-sampling=' + str(self.label_sampling)
        if self.instance_sampling is not None:
            name += '_instance-sampling=' + str(self.instance_sampling)
        if self.feature_sampling is not None:
            name += '_feature-sampling=' + str(self.feature_sampling)
        if self.holdout is not None:
            name += '_holdout=' + str(self.holdout)
        if self.feature_binning is not None:
            name += '_feature-binning=' + str(self.feature_binning)
        if self.label_binning is not None and self.label_binning != AUTOMATIC:
            name += '_label-binning=' + str(self.label_binning)
        if self.pruning is not None:
            name += '_pruning=' + str(self.pruning)
        if float(self.shrinkage) < 1.0:
            name += '_shrinkage=' + str(self.shrinkage)
        if float(self.l1_regularization_weight) > 0.0:
            name += '_l1=' + str(self.l1_regularization_weight)
        if float(self.l2_regularization_weight) > 0.0:
            name += '_l2=' + str(self.l2_regularization_weight)
        if int(self.min_coverage) > 1:
            name += '_min-coverage=' + str(self.min_coverage)
        if int(self.max_conditions) > 0:
            name += '_max-conditions=' + str(self.max_conditions)
        if int(self.max_head_refinements) > 0:
            name += '_max-head-refinements=' + str(self.max_head_refinements)
        if int(self.random_state) != 1:
            name += '_random_state=' + str(self.random_state)
        return name

    def _create_statistics_provider_factory(self, feature_characteristics: FeatureCharacteristics,
                                            label_characteristics: LabelCharacteristics) -> StatisticsProviderFactory:
        head_type = parse_param("head_type", self.__get_preferred_head_type(), HEAD_TYPE_VALUES)
        default_rule_head_type = HEAD_TYPE_COMPLETE if self._use_default_rule(feature_characteristics,
                                                                              label_characteristics) else head_type
        num_threads = create_num_threads(self.__get_preferred_parallel_statistic_update(label_characteristics),
                                         'parallel_statistic_update')
        loss_function = self.__create_loss_function()
        evaluation_measure = self.__create_loss_function()
        label_binning_factory = self.__create_label_binning_factory()

        if label_binning_factory is not None and head_type == HEAD_TYPE_SINGLE:
            log.warning('Parameter "label_binning" does not have any effect, because parameter "head_type" is set to "'
                        + self.head_type + '"!')

        default_rule_evaluation_factory = self.__create_rule_evaluation_factory(loss_function, default_rule_head_type,
                                                                                label_binning_factory)
        regular_rule_evaluation_factory = self.__create_rule_evaluation_factory(loss_function, head_type,
                                                                                self.__create_label_binning_factory())
        pruning_rule_evaluation_factory = self.__create_rule_evaluation_factory(loss_function, head_type,
                                                                                self.__create_label_binning_factory())

        if isinstance(loss_function, LabelWiseLoss):
            return DenseLabelWiseStatisticsProviderFactory(loss_function, evaluation_measure,
                                                           default_rule_evaluation_factory,
                                                           regular_rule_evaluation_factory,
                                                           pruning_rule_evaluation_factory, num_threads)
        else:
            if head_type == HEAD_TYPE_SINGLE:
                return DenseConvertibleExampleWiseStatisticsProviderFactory(loss_function, evaluation_measure,
                                                                            default_rule_evaluation_factory,
                                                                            regular_rule_evaluation_factory,
                                                                            pruning_rule_evaluation_factory,
                                                                            num_threads)
            else:
                return DenseExampleWiseStatisticsProviderFactory(loss_function, evaluation_measure,
                                                                 default_rule_evaluation_factory,
                                                                 regular_rule_evaluation_factory,
                                                                 pruning_rule_evaluation_factory, num_threads)

    def _create_thresholds_factory(self, feature_characteristics: FeatureCharacteristics,
                                   label_characteristics: LabelCharacteristics) -> ThresholdsFactory:
        num_threads = create_num_threads(self.__get_preferred_parallel_statistic_update(label_characteristics),
                                         'parallel_statistic_update')
        return create_thresholds_factory(self.feature_binning, num_threads)

    def _create_rule_induction(self, feature_characteristics: FeatureCharacteristics,
                               label_characteristics: LabelCharacteristics) -> RuleInduction:
        num_threads = create_num_threads(self.__get_preferred_parallel_rule_refinement(feature_characteristics),
                                         'parallel_rule_refinement')
        return TopDownRuleInduction(int(self.min_coverage), int(self.max_conditions), int(self.max_head_refinements),
                                    BooleanOption.parse(self.recalculate_predictions), num_threads)

    def _create_rule_model_assemblage_factory(
            self, feature_characteristics: FeatureCharacteristics,
            label_characteristics: LabelCharacteristics) -> RuleModelAssemblageFactory:
        return SequentialRuleModelAssemblageFactory()

    def _create_label_sampling_factory(self, feature_characteristics: FeatureCharacteristics,
                                       label_characteristics: LabelCharacteristics) -> Optional[LabelSamplingFactory]:
        return create_label_sampling_factory(self.label_sampling)

    def _create_instance_sampling_factory(
            self, feature_characteristics: FeatureCharacteristics,
            label_characteristics: LabelCharacteristics) -> Optional[InstanceSamplingFactory]:
        return create_instance_sampling_factory(self.instance_sampling)

    def _create_feature_sampling_factory(
            self, feature_characteristics: FeatureCharacteristics,
            label_characteristics: LabelCharacteristics) -> Optional[FeatureSamplingFactory]:
        return create_feature_sampling_factory(self.feature_sampling)

    def _create_partition_sampling_factory(
            self, feature_characteristics: FeatureCharacteristics,
            label_characteristics: LabelCharacteristics) -> Optional[PartitionSamplingFactory]:
        return create_partition_sampling_factory(self.holdout)

    def _create_pruning(self, feature_characteristics: FeatureCharacteristics,
                        label_characteristics: LabelCharacteristics) -> Optional[Pruning]:
        return create_pruning(self.pruning, self.instance_sampling)

    def _create_post_processor(self, feature_characteristics: FeatureCharacteristics,
                               label_characteristics: LabelCharacteristics) -> Optional[PostProcessor]:
        shrinkage = float(self.shrinkage)

        if shrinkage == 1.0:
            return NoPostProcessor()
        else:
            return ConstantShrinkage(shrinkage)

    def _create_stopping_criteria(self, feature_characteristics: FeatureCharacteristics,
                                  label_characteristics: LabelCharacteristics) -> List[StoppingCriterion]:
        stopping_criteria = create_stopping_criteria(int(self.max_rules), int(self.time_limit))
        early_stopping_criterion = self.__create_early_stopping()

        if early_stopping_criterion is not None:
            stopping_criteria.append(early_stopping_criterion)

        return stopping_criteria

    def _use_default_rule(self, feature_characteristics: FeatureCharacteristics,
                          label_characteristics: LabelCharacteristics) -> bool:
        return BooleanOption.parse(self.default_rule)

    def __create_early_stopping(self) -> Optional[MeasureStoppingCriterion]:
        early_stopping = self.early_stopping

        if early_stopping is not None:
            value, options = parse_param_and_options('early_stopping', early_stopping, EARLY_STOPPING_VALUES)

            if value == EARLY_STOPPING_LOSS:
                if self.holdout is None:
                    log.warning('Parameter "early_stopping" does not have any effect, because parameter "holdout" is '
                                + 'set to "None"!')
                    return None
                else:
                    aggregation_function = self.__create_aggregation_function(
                        options.get_string(ARGUMENT_AGGREGATION_FUNCTION, 'avg'))
                    min_rules = options.get_int(ARGUMENT_MIN_RULES, 100)
                    update_interval = options.get_int(ARGUMENT_UPDATE_INTERVAL, 1)
                    stop_interval = options.get_int(ARGUMENT_STOP_INTERVAL, 1)
                    num_past = options.get_int(ARGUMENT_NUM_PAST, 50)
                    num_recent = options.get_int(ARGUMENT_NUM_RECENT, 50)
                    min_improvement = options.get_float(ARGUMENT_MIN_IMPROVEMENT, 0.005)
                    force_stop = options.get_bool(ARGUMENT_FORCE_STOP, True)
                    return MeasureStoppingCriterion(aggregation_function, min_rules=min_rules,
                                                    update_interval=update_interval, stop_interval=stop_interval,
                                                    num_past=num_past, num_recent=num_recent,
                                                    min_improvement=min_improvement, force_stop=force_stop)
        return None

    @staticmethod
    def __create_aggregation_function(aggregation_function: str) -> AggregationFunction:
        value = parse_param(ARGUMENT_AGGREGATION_FUNCTION, aggregation_function, {AGGREGATION_FUNCTION_MIN,
                                                                                  AGGREGATION_FUNCTION_MAX,
                                                                                  AGGREGATION_FUNCTION_ARITHMETIC_MEAN})
        if value == AGGREGATION_FUNCTION_MIN:
            return MinFunction()
        elif value == AGGREGATION_FUNCTION_MAX:
            return MaxFunction()
        elif value == AGGREGATION_FUNCTION_ARITHMETIC_MEAN:
            return ArithmeticMeanFunction()

    def __create_loss_function(self):
        value = parse_param("loss", self.loss, LOSS_VALUES)

        if value == LOSS_SQUARED_ERROR_LABEL_WISE:
            return LabelWiseSquaredErrorLoss()
        elif value == LOSS_SQUARED_HINGE_LABEL_WISE:
            return LabelWiseSquaredHingeLoss()
        elif value == LOSS_LOGISTIC_LABEL_WISE:
            return LabelWiseLogisticLoss()
        elif value == LOSS_LOGISTIC_EXAMPLE_WISE:
            return ExampleWiseLogisticLoss()

    def __create_rule_evaluation_factory(self, loss_function, head_type: str,
                                         label_binning_factory: LabelBinningFactory):
        l1_regularization_weight = float(self.l1_regularization_weight)
        l2_regularization_weight = float(self.l2_regularization_weight)

        if head_type == HEAD_TYPE_SINGLE:
            return LabelWiseSingleLabelRuleEvaluationFactory(l1_regularization_weight, l2_regularization_weight)
        elif head_type == HEAD_TYPE_COMPLETE:
            if isinstance(loss_function, LabelWiseLoss):
                if label_binning_factory is None:
                    return LabelWiseCompleteRuleEvaluationFactory(l1_regularization_weight, l2_regularization_weight)
                else:
                    return LabelWiseCompleteBinnedRuleEvaluationFactory(l1_regularization_weight,
                                                                        l2_regularization_weight, label_binning_factory)
            else:
                if label_binning_factory is None:
                    return ExampleWiseCompleteRuleEvaluationFactory(l1_regularization_weight, l2_regularization_weight)
                else:
                    return ExampleWiseCompleteBinnedRuleEvaluationFactory(l1_regularization_weight,
                                                                          l2_regularization_weight,
                                                                          label_binning_factory)

        raise ValueError('configuration currently not supported :-(')

    def __create_label_binning_factory(self) -> LabelBinningFactory:
        label_binning = self.__get_preferred_label_binning()

        if label_binning is not None:
            value, options = parse_param_and_options('label_binning', label_binning, LABEL_BINNING_VALUES)

            if value == LABEL_BINNING_EQUAL_WIDTH:
                bin_ratio = options.get_float(ARGUMENT_BIN_RATIO, 0.04)
                min_bins = options.get_int(ARGUMENT_MIN_BINS, 1)
                max_bins = options.get_int(ARGUMENT_MAX_BINS, 0)
                return EqualWidthLabelBinningFactory(bin_ratio, min_bins, max_bins)
        return None

    def __get_preferred_label_binning(self) -> Optional[str]:
        label_binning = self.label_binning

        if label_binning == AUTOMATIC:
            if self.loss in NON_DECOMPOSABLE_LOSSES and self.__get_preferred_head_type() == HEAD_TYPE_COMPLETE:
                return LABEL_BINNING_EQUAL_WIDTH
            else:
                return None
        return label_binning

    def __get_preferred_head_type(self) -> str:
        head_type = self.head_type

        if head_type == AUTOMATIC:
            if self.loss in NON_DECOMPOSABLE_LOSSES:
                return HEAD_TYPE_COMPLETE
            else:
                return HEAD_TYPE_SINGLE
        return head_type

    def __get_preferred_parallel_rule_refinement(self, feature_characteristics: FeatureCharacteristics) -> str:
        parallel_rule_refinement = self.parallel_rule_refinement

        if parallel_rule_refinement == AUTOMATIC:
            head_type = self.__get_preferred_head_type()

            if head_type != HEAD_TYPE_SINGLE and self.loss in NON_DECOMPOSABLE_LOSSES:
                return BooleanOption.FALSE.value
            else:
                if feature_characteristics.is_sparse() and self.feature_sampling is not None:
                    return BooleanOption.FALSE.value
                else:
                    return BooleanOption.TRUE.value
        return parallel_rule_refinement

    def __get_preferred_parallel_statistic_update(self, label_characteristics: LabelCharacteristics) -> str:
        parallel_statistic_update = self.parallel_statistic_update

        if parallel_statistic_update == AUTOMATIC:
            if self.loss in NON_DECOMPOSABLE_LOSSES and label_characteristics.get_num_labels() >= 20:
                return BooleanOption.TRUE.value
            else:
                return BooleanOption.FALSE.value
        return parallel_statistic_update

    def _create_model_builder(self, feature_characteristics: FeatureCharacteristics,
                              label_characteristics: LabelCharacteristics) -> ModelBuilder:
        return RuleListBuilder()

    def _create_predictor(self, feature_characteristics: FeatureCharacteristics,
                          label_characteristics: LabelCharacteristics) -> Predictor:
        predictor = self.__get_preferred_predictor()
        value = parse_param('predictor', predictor, PREDICTOR_VALUES)

        if value == PREDICTOR_LABEL_WISE:
            return self.__create_label_wise_predictor(label_characteristics)
        elif value == PREDICTOR_EXAMPLE_WISE:
            return self.__create_example_wise_predictor(label_characteristics)

    def _create_probability_predictor(self, feature_characteristics: FeatureCharacteristics,
                                      label_characteristics: LabelCharacteristics) -> Optional[Predictor]:
        predictor = self.__get_preferred_predictor()

        if self.loss == LOSS_LOGISTIC_LABEL_WISE or self.loss == LOSS_LOGISTIC_EXAMPLE_WISE:
            if predictor == PREDICTOR_LABEL_WISE:
                transformation_function = LogisticFunction()
                return self.__create_label_wise_probability_predictor(transformation_function, label_characteristics)
        return None

    def _create_label_vector_set(self, label_matrix: LabelMatrix) -> Optional[LabelVectorSet]:
        predictor = self.__get_preferred_predictor()

        if predictor == PREDICTOR_EXAMPLE_WISE:
            return LabelVectorSet.create(label_matrix)
        return None

    def __get_preferred_predictor(self) -> str:
        predictor = self.predictor

        if predictor == AUTOMATIC:
            if self.loss in NON_DECOMPOSABLE_LOSSES:
                return PREDICTOR_EXAMPLE_WISE
            else:
                return PREDICTOR_LABEL_WISE
        return predictor

    def __create_label_wise_predictor(self,
                                      label_characteristics: LabelCharacteristics) -> LabelWiseClassificationPredictor:
        num_threads = create_num_threads(self.parallel_prediction, 'parallel_prediction')
        threshold = 0.5 if self.loss == LOSS_SQUARED_HINGE_LABEL_WISE else 0.0
        return LabelWiseClassificationPredictor(num_labels=label_characteristics.get_num_labels(), threshold=threshold,
                                                num_threads=num_threads)

    def __create_example_wise_predictor(
            self, label_characteristics: LabelCharacteristics) -> ExampleWiseClassificationPredictor:
        loss = self.__create_loss_function()
        num_threads = create_num_threads(self.parallel_prediction, 'parallel_prediction')
        return ExampleWiseClassificationPredictor(num_labels=label_characteristics.get_num_labels(), measure=loss,
                                                  num_threads=num_threads)

    def __create_label_wise_probability_predictor(
            self, transformation_function: LabelWiseTransformationFunction,
            label_characteristics: LabelCharacteristics) -> LabelWiseProbabilityPredictor:
        num_threads = create_num_threads(self.parallel_prediction, 'parallel_prediction')
        return LabelWiseProbabilityPredictor(num_labels=label_characteristics.get_num_labels(),
                                             transformation_function=transformation_function, num_threads=num_threads)
