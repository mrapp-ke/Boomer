#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides scikit-learn implementations of separate-and-conquer algorithms.
"""
from typing import Dict, Set, Optional, List

from mlrl.common.cython.feature_sampling import FeatureSamplingFactory
from mlrl.common.cython.instance_sampling import InstanceSamplingFactory
from mlrl.common.cython.label_sampling import LabelSamplingFactory
from mlrl.common.cython.model import ModelBuilder
from mlrl.common.cython.output import Predictor
from mlrl.common.cython.partition_sampling import PartitionSamplingFactory
from mlrl.common.cython.pruning import Pruning
from mlrl.common.cython.rule_induction import RuleInduction, TopDownRuleInduction
from mlrl.common.cython.rule_model_assemblage import RuleModelAssemblageFactory, SequentialRuleModelAssemblageFactory
from mlrl.common.cython.statistics import StatisticsProviderFactory
from mlrl.common.cython.stopping import StoppingCriterion
from mlrl.common.cython.thresholds import ThresholdsFactory
from mlrl.seco.cython.heuristics import Heuristic, Accuracy, Precision, Recall, Laplace, WRA, FMeasure, MEstimate
from mlrl.seco.cython.instance_sampling import InstanceSamplingWithReplacementFactory, \
    InstanceSamplingWithoutReplacementFactory
from mlrl.seco.cython.model import DecisionListBuilder
from mlrl.seco.cython.output import LabelWiseClassificationPredictor
from mlrl.seco.cython.rule_evaluation_label_wise import LiftFunction, PeakLiftFunction, \
    LabelWiseMajorityRuleEvaluationFactory, LabelWisePartialRuleEvaluationFactory, \
    LabelWiseSingleLabelRuleEvaluationFactory
from mlrl.seco.cython.statistics_label_wise import DenseLabelWiseStatisticsProviderFactory
from mlrl.seco.cython.stopping import CoverageStoppingCriterion
from sklearn.base import ClassifierMixin

from mlrl.common.options import BooleanOption
from mlrl.common.rule_learners import HEAD_TYPE_SINGLE, PRUNING_IREP, SAMPLING_WITH_REPLACEMENT, \
    SAMPLING_WITHOUT_REPLACEMENT, ARGUMENT_SAMPLE_SIZE
from mlrl.common.rule_learners import MLRuleLearner, SparsePolicy
from mlrl.common.rule_learners import create_pruning, create_feature_sampling_factory, \
    create_label_sampling_factory, create_partition_sampling_factory, create_stopping_criteria, \
    create_num_threads, create_thresholds_factory, parse_param_and_options, parse_param

HEAD_TYPE_PARTIAL = 'partial'

HEURISTIC_ACCURACY = 'accuracy'

HEURISTIC_PRECISION = 'precision'

HEURISTIC_LAPLACE = 'laplace'

HEURISTIC_RECALL = 'recall'

HEURISTIC_WRA = 'weighted-relative-accuracy'

HEURISTIC_F_MEASURE = 'f-measure'

HEURISTIC_M_ESTIMATE = 'm-estimate'

LIFT_FUNCTION_PEAK = 'peak'

ARGUMENT_PEAK_LABEL = 'peak_label'

ARGUMENT_MAX_LIFT = 'max_lift'

ARGUMENT_CURVATURE = 'curvature'

ARGUMENT_BETA = 'beta'

ARGUMENT_M = 'm'

INSTANCE_SAMPLING_VALUES: Dict[str, Set[str]] = {
    SAMPLING_WITH_REPLACEMENT: {ARGUMENT_SAMPLE_SIZE},
    SAMPLING_WITHOUT_REPLACEMENT: {ARGUMENT_SAMPLE_SIZE}
}

HEAD_TYPE_VALUES: Set[str] = {HEAD_TYPE_SINGLE, HEAD_TYPE_PARTIAL}

HEURISTIC_VALUES: Dict[str, Set[str]] = {
    HEURISTIC_ACCURACY: {},
    HEURISTIC_PRECISION: {},
    HEURISTIC_RECALL: {},
    HEURISTIC_LAPLACE: {},
    HEURISTIC_WRA: {},
    HEURISTIC_F_MEASURE: {ARGUMENT_BETA},
    HEURISTIC_M_ESTIMATE: {ARGUMENT_M}
}

LIFT_FUNCTION_VALUES: Dict[str, Set[str]] = {
    LIFT_FUNCTION_PEAK: {ARGUMENT_PEAK_LABEL, ARGUMENT_MAX_LIFT, ARGUMENT_CURVATURE}
}


class SeCoRuleLearner(MLRuleLearner, ClassifierMixin):
    """
    A scikit-multilearn implementation of an Separate-and-Conquer (SeCo) algorithm for learning multi-label
    classification rules.
    """

    def __init__(self, random_state: int = 1, feature_format: str = SparsePolicy.AUTO.value,
                 label_format: str = SparsePolicy.AUTO.value, prediction_format: str = SparsePolicy.AUTO.value,
                 max_rules: int = 500, time_limit: int = 0, head_type: str = HEAD_TYPE_SINGLE,
                 lift_function: str = LIFT_FUNCTION_PEAK, heuristic: str = HEURISTIC_F_MEASURE,
                 pruning_heuristic: str = HEURISTIC_ACCURACY, label_sampling: str = None,
                 instance_sampling: str = SAMPLING_WITHOUT_REPLACEMENT, feature_sampling: str = None,
                 holdout: str = None, feature_binning: str = None, pruning: str = PRUNING_IREP, min_coverage: int = 1,
                 max_conditions: int = 0, max_head_refinements: int = 1,
                 parallel_rule_refinement: str = BooleanOption.TRUE.value,
                 parallel_statistic_update: str = BooleanOption.FALSE.value,
                 parallel_prediction: str = BooleanOption.TRUE.value):
        """
        :param max_rules:                           The maximum number of rules to be induced (including the default
                                                    rule)
        :param time_limit:                          The duration in seconds after which the induction of rules should be
                                                    canceled
        :param head_type:                           The type of the rule heads that should be used. Must be
                                                    `single-label` or `partial`
        :param lift_function:                       The lift function to use. Must be `peak`. Additional options may be
                                                    provided using the bracket notation
                                                    `peak{peak_label=10,max_lift=2.0,curvature=1.0}`
        :param heuristic:                           The heuristic to be minimized. Must be `accuracy`, `precision`,
                                                    `recall`, `weighted-relative-accuracy`, `f-measure`, `m-estimate` or
                                                    `laplace`. Additional options may be provided using the bracket
                                                    notation `f-measure{beta=1.0}`
        :param pruning_heuristic:                   The heuristic to be used for pruning. Must be `accuracy`,
                                                    `precision`, `recall`, `weighted-relative-accuracy`, `f-measure`,
                                                    `m-estimate` or `laplace`. Additional options may be provided using
                                                    the bracket notation `f-measure{beta=1.0}`
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
        :param pruning:                             The strategy that is used for pruning rules. Must be `irep` or None,
                                                    if no pruning should be used
        :param min_coverage:                        The minimum number of training examples that must be covered by a
                                                    rule. Must be at least 1
        :param max_conditions:                      The maximum number of conditions to be included in a rule's body.
                                                    Must be at least 1 or 0, if the number of conditions should not be
                                                    restricted
        :param max_head_refinements:                The maximum number of times the head of a rule may be refined after
                                                    a new condition has been added to its body. Must be at least 1 or
                                                    0, if the number of refinements should not be restricted
        :param parallel_rule_refinement:            Whether potential refinements of rules should be searched for in
                                                    parallel or not. Must be `true` or `false`. Additional options may
                                                    be provided using the bracket notation `true{num_threads=8}`
        :param parallel_statistic_update:           Whether the confusion matrices for different examples should be
                                                    calculated in parallel or not. Must be `true` or `false`. Additional
                                                    options may be provided using the bracket notation
                                                    `true{num_threads=8}`
        :param parallel_prediction:                 Whether predictions for different examples should be obtained in
                                                    parallel or not. Must be `true` or `false`. Additional options may
                                                    be provided using the bracket notation `true{num_threads=8}`
        """
        super().__init__(random_state, feature_format, label_format, prediction_format)
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.head_type = head_type
        self.lift_function = lift_function
        self.heuristic = heuristic
        self.pruning_heuristic = pruning_heuristic
        self.label_sampling = label_sampling
        self.instance_sampling = instance_sampling
        self.feature_sampling = feature_sampling
        self.holdout = holdout
        self.feature_binning = feature_binning
        self.pruning = pruning
        self.min_coverage = min_coverage
        self.max_conditions = max_conditions
        self.max_head_refinements = max_head_refinements
        self.parallel_rule_refinement = parallel_rule_refinement
        self.parallel_statistic_update = parallel_statistic_update
        self.parallel_prediction = parallel_prediction

    def get_name(self) -> str:
        name = 'max-rules=' + str(self.max_rules)
        name += '_head-type=' + str(self.head_type)
        name += '_lift-function=' + str(self.lift_function)
        name += '_heuristic=' + str(self.heuristic)
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
        if self.pruning is not None:
            name += '_pruning-heuristic=' + str(self.pruning_heuristic)
            name += '_pruning=' + str(self.pruning)
        if int(self.min_coverage) > 1:
            name += '_min-coverage=' + str(self.min_coverage)
        if int(self.max_conditions) > 0:
            name += '_max-conditions=' + str(self.max_conditions)
        if int(self.max_head_refinements) > 0:
            name += '_max-head-refinements=' + str(self.max_head_refinements)
        if int(self.random_state) != 1:
            name += '_random_state=' + str(self.random_state)
        return name

    def _create_statistics_provider_factory(self, num_labels: int) -> StatisticsProviderFactory:
        heuristic = self.__create_heuristic(self.heuristic, 'heuristic')
        pruning_heuristic = self.__create_heuristic(self.pruning_heuristic, 'pruning_heuristic')
        head_type = parse_param('head_type', self.head_type, HEAD_TYPE_VALUES)
        default_rule_evaluation_factory = LabelWiseMajorityRuleEvaluationFactory()
        regular_rule_evaluation_factory = self.__create_rule_evaluation_factory(head_type, heuristic,
                                                                                num_labels)
        pruning_rule_evaluation_factory = self.__create_rule_evaluation_factory(head_type, pruning_heuristic,
                                                                                num_labels)
        return DenseLabelWiseStatisticsProviderFactory(default_rule_evaluation_factory, regular_rule_evaluation_factory,
                                                       pruning_rule_evaluation_factory)

    def _create_thresholds_factory(self) -> ThresholdsFactory:
        num_threads = create_num_threads(self.parallel_statistic_update, 'parallel_statistic_update')
        return create_thresholds_factory(self.feature_binning, num_threads)

    def _create_rule_induction(self) -> RuleInduction:
        num_threads = create_num_threads(self.parallel_rule_refinement, 'parallel_rule_refinement')
        return TopDownRuleInduction(int(self.min_coverage), int(self.max_conditions), int(self.max_head_refinements),
                                    False, num_threads)

    def _create_rule_model_assemblage_factory(self) -> RuleModelAssemblageFactory:
        return SequentialRuleModelAssemblageFactory()

    def _create_label_sampling_factory(self) -> Optional[LabelSamplingFactory]:
        return create_label_sampling_factory(self.label_sampling)

    def _create_instance_sampling_factory(self) -> Optional[InstanceSamplingFactory]:
        instance_sampling = self.instance_sampling

        if instance_sampling is not None:
            value, options = parse_param_and_options('instance_sampling', instance_sampling, INSTANCE_SAMPLING_VALUES)

            if value == SAMPLING_WITH_REPLACEMENT:
                sample_size = options.get_float(ARGUMENT_SAMPLE_SIZE, 1.0)
                return InstanceSamplingWithReplacementFactory(sample_size)
            elif value == SAMPLING_WITHOUT_REPLACEMENT:
                sample_size = options.get_float(ARGUMENT_SAMPLE_SIZE, 0.66)
                return InstanceSamplingWithoutReplacementFactory(sample_size)
        return None

    def _create_feature_sampling_factory(self) -> Optional[FeatureSamplingFactory]:
        return create_feature_sampling_factory(self.feature_sampling)

    def _create_partition_sampling_factory(self) -> Optional[PartitionSamplingFactory]:
        return create_partition_sampling_factory(self.holdout)

    def _create_pruning(self) -> Optional[Pruning]:
        return create_pruning(self.pruning, self.instance_sampling)

    def _create_stopping_criteria(self) -> List[StoppingCriterion]:
        stopping_criteria = create_stopping_criteria(int(self.max_rules), int(self.time_limit))
        stopping_criteria.append(CoverageStoppingCriterion(0))
        return stopping_criteria

    @staticmethod
    def __create_heuristic(heuristic: str, parameter_name: str) -> Heuristic:
        value, options = parse_param_and_options(parameter_name, heuristic, HEURISTIC_VALUES)

        if value == HEURISTIC_ACCURACY:
            return Accuracy()
        elif value == HEURISTIC_PRECISION:
            return Precision()
        elif value == HEURISTIC_RECALL:
            return Recall()
        elif value == HEURISTIC_LAPLACE:
            return Laplace()
        elif value == HEURISTIC_WRA:
            return WRA()
        elif value == HEURISTIC_F_MEASURE:
            beta = options.get_float(ARGUMENT_BETA, 0.25)
            return FMeasure(beta)
        elif value == HEURISTIC_M_ESTIMATE:
            m = options.get_float(ARGUMENT_M, 22.466)
            return MEstimate(m)

    def __create_lift_function(self, num_labels: int) -> LiftFunction:
        value, options = parse_param_and_options('lift_function', self.lift_function, LIFT_FUNCTION_VALUES)

        if value == LIFT_FUNCTION_PEAK:
            peak_label = options.get_int(ARGUMENT_PEAK_LABEL, int(num_labels / 2) + 1)
            max_lift = options.get_float(ARGUMENT_MAX_LIFT, 1.5)
            curvature = options.get_float(ARGUMENT_CURVATURE, 1.0)
            return PeakLiftFunction(num_labels, peak_label, max_lift, curvature)

    def __create_rule_evaluation_factory(self, head_type: str, heuristic: Heuristic, num_labels: int):
        if head_type == HEAD_TYPE_SINGLE:
            return LabelWiseSingleLabelRuleEvaluationFactory(heuristic)
        else:
            return LabelWisePartialRuleEvaluationFactory(heuristic, self.__create_lift_function(num_labels))

    def _create_model_builder(self) -> ModelBuilder:
        return DecisionListBuilder()

    def _create_predictor(self, num_labels: int) -> Predictor:
        return self.__create_label_wise_predictor(num_labels)

    def __create_label_wise_predictor(self, num_labels: int) -> LabelWiseClassificationPredictor:
        num_threads = create_num_threads(self.parallel_prediction, 'parallel_prediction')
        return LabelWiseClassificationPredictor(num_labels, num_threads)
