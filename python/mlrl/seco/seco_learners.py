#!/usr/bin/python

from mlrl.common.cython.head_refinement import HeadRefinementFactory, SingleLabelHeadRefinementFactory, \
    FullHeadRefinementFactory
from mlrl.common.cython.input import CContiguousLabelMatrix
from mlrl.common.cython.model import ModelBuilder
from mlrl.common.cython.output import Predictor
from mlrl.common.cython.post_processing import NoPostProcessor
from mlrl.common.cython.rule_induction import TopDownRuleInduction, SequentialRuleModelInduction
from mlrl.common.cython.statistics import StatisticsProviderFactory
from mlrl.seco.cython.head_refinement import PartialHeadRefinementFactory, LiftFunction, PeakLiftFunction
from mlrl.seco.cython.heuristics import Heuristic, Precision, Recall, WRA, HammingLoss, FMeasure, MEstimate
from mlrl.seco.cython.model import DecisionListBuilder
from mlrl.seco.cython.output import LabelWiseClassificationPredictor
from mlrl.seco.cython.rule_evaluation_label_wise import HeuristicLabelWiseRuleEvaluationFactory
from mlrl.seco.cython.statistics_label_wise import LabelWiseStatisticsProviderFactory
from mlrl.seco.cython.stopping import CoverageStoppingCriterion
from sklearn.base import ClassifierMixin

from mlrl.common.rule_learners import HEAD_REFINEMENT_SINGLE
from mlrl.common.rule_learners import MLRuleLearner, SparsePolicy
from mlrl.common.rule_learners import create_pruning, create_feature_sub_sampling, create_instance_sub_sampling, \
    create_label_sub_sampling, create_partition_sampling, create_max_conditions, create_stopping_criteria, \
    create_min_coverage, create_max_head_refinements, get_preferred_num_threads, parse_prefix_and_dict, \
    get_int_argument, get_float_argument, create_thresholds_factory

HEAD_REFINEMENT_PARTIAL = 'partial'

AVERAGING_LABEL_WISE = 'label-wise-averaging'

HEURISTIC_PRECISION = 'precision'

HEURISTIC_HAMMING_LOSS = 'hamming-loss'

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


class SeparateAndConquerRuleLearner(MLRuleLearner, ClassifierMixin):
    """
    A scikit-multilearn implementation of an Separate-and-Conquer algorithm for learning multi-label classification
    rules.
    """

    def __init__(self, random_state: int = 1, feature_format: str = SparsePolicy.AUTO.value,
                 label_format: str = SparsePolicy.AUTO.value, max_rules: int = 500, time_limit: int = -1,
                 head_refinement: str = None, lift_function: str = LIFT_FUNCTION_PEAK, loss: str = AVERAGING_LABEL_WISE,
                 heuristic: str = HEURISTIC_PRECISION, label_sub_sampling: str = None,
                 instance_sub_sampling: str = None, feature_sub_sampling: str = None, holdout_set_size: float = 0.0,
                 feature_binning: str = None, pruning: str = None, min_coverage: int = 1, max_conditions: int = -1,
                 max_head_refinements: int = 1, num_threads_refinement: int = 1, num_threads_update: int = 1,
                 num_threads_prediction: int = 1):
        """
        :param max_rules:                           The maximum number of rules to be induced (including the default
                                                    rule)
        :param time_limit:                          The duration in seconds after which the induction of rules should be
                                                    canceled
        :param head_refinement:                     The strategy that is used to find the heads of rules. Must be
                                                    `single-label`, `partial` or None, if the default strategy should be
                                                    used
        :param lift_function:                       The lift function to use. Must be `peak`. Additional arguments may
                                                    be provided as a dictionary, e.g.
                                                    `peak{\"peak_label\":10,\"max_lift\":2.0,\"curvature\":1.0}`
        :param loss:                                The loss function to be minimized. Must be `label-wise-averaging`
        :param heuristic:                           The heuristic to be minimized. Must be `precision`, `hamming-loss`,
                                                    `recall`, `weighted-relative-accuracy`, `f-measure` or `m-estimate`.
                                                    Additional arguments may be provided as a dictionary, e.g.
                                                    `f-measure{\"beta\":1.0}`
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
        :param pruning:                             The strategy that is used for pruning rules. Must be `irep` or None,
                                                    if no pruning should be used
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
        self.head_refinement = head_refinement
        self.lift_function = lift_function
        self.loss = loss
        self.heuristic = heuristic
        self.label_sub_sampling = label_sub_sampling
        self.instance_sub_sampling = instance_sub_sampling
        self.feature_sub_sampling = feature_sub_sampling
        self.holdout_set_size = holdout_set_size
        self.feature_binning = feature_binning
        self.pruning = pruning
        self.min_coverage = min_coverage
        self.max_conditions = max_conditions
        self.max_head_refinements = max_head_refinements
        self.num_threads_refinement = num_threads_refinement
        self.num_threads_update = num_threads_update
        self.num_threads_prediction = num_threads_prediction

    def get_name(self) -> str:
        name = 'max-rules=' + str(self.max_rules)
        if self.head_refinement is not None:
            name += '_head-refinement=' + str(self.head_refinement)
        name += '_lift-function=' + str(self.lift_function)
        name += '_loss=' + str(self.loss)
        name += '_heuristic=' + str(self.heuristic)
        if self.label_sub_sampling is not None:
            name += '_label-sub-sampling=' + str(self.label_sub_sampling)
        if self.instance_sub_sampling is not None:
            name += '_instance-sub-sampling=' + str(self.instance_sub_sampling)
        if self.feature_sub_sampling is not None:
            name += '_feature-sub-sampling=' + str(self.feature_sub_sampling)
        if float(self.holdout_set_size > 0.0):
            name += '_holdout=' + str(self.holdout_set_size)
        if self.feature_binning is not None:
            name += '_feature-binning=' + str(self.feature_binning)
        if self.pruning is not None:
            name += '_pruning=' + str(self.pruning)
        if int(self.min_coverage) > 1:
            name += '_min-coverage=' + str(self.min_coverage)
        if int(self.max_conditions) != -1:
            name += '_max-conditions=' + str(self.max_conditions)
        if int(self.max_head_refinements) != -1:
            name += '_max-head-refinements=' + str(self.max_head_refinements)
        if int(self.random_state) != 1:
            name += '_random_state=' + str(self.random_state)
        return name

    def _create_model_builder(self) -> ModelBuilder:
        return DecisionListBuilder()

    def _create_rule_model_induction(self, num_labels: int) -> SequentialRuleModelInduction:
        heuristic = self.__create_heuristic()
        statistics_provider_factory = self.__create_statistics_provider_factory(heuristic)
        num_threads_update = get_preferred_num_threads(self.num_threads_update)
        thresholds_factory = create_thresholds_factory(self.feature_binning, num_threads_update)
        num_threads_refinement = get_preferred_num_threads(self.num_threads_refinement)
        rule_induction = TopDownRuleInduction(num_threads_refinement)
        lift_function = self.__create_lift_function(num_labels)
        default_rule_head_refinement_factory = FullHeadRefinementFactory()
        head_refinement_factory = self.__create_head_refinement_factory(lift_function)
        label_sub_sampling = create_label_sub_sampling(self.label_sub_sampling, num_labels)
        instance_sub_sampling = create_instance_sub_sampling(self.instance_sub_sampling)
        feature_sub_sampling = create_feature_sub_sampling(self.feature_sub_sampling)
        partition_sampling = create_partition_sampling(self.holdout_set_size)
        pruning = create_pruning(self.pruning, self.instance_sub_sampling)
        post_processor = NoPostProcessor()
        min_coverage = create_min_coverage(self.min_coverage)
        max_conditions = create_max_conditions(self.max_conditions)
        max_head_refinements = create_max_head_refinements(self.max_head_refinements)
        stopping_criteria = create_stopping_criteria(int(self.max_rules), int(self.time_limit))
        stopping_criteria.append(CoverageStoppingCriterion(0))
        return SequentialRuleModelInduction(statistics_provider_factory, thresholds_factory, rule_induction,
                                            default_rule_head_refinement_factory, head_refinement_factory,
                                            label_sub_sampling, instance_sub_sampling, feature_sub_sampling,
                                            partition_sampling, pruning, post_processor, min_coverage, max_conditions,
                                            max_head_refinements, stopping_criteria)

    def __create_heuristic(self) -> Heuristic:
        heuristic = self.heuristic
        prefix, args = parse_prefix_and_dict(heuristic, [HEURISTIC_PRECISION, HEURISTIC_HAMMING_LOSS, HEURISTIC_RECALL,
                                                         HEURISTIC_WRA, HEURISTIC_F_MEASURE, HEURISTIC_M_ESTIMATE])

        if prefix == HEURISTIC_PRECISION:
            return Precision()
        elif prefix == HEURISTIC_HAMMING_LOSS:
            return HammingLoss()
        elif prefix == HEURISTIC_RECALL:
            return Recall()
        elif prefix == HEURISTIC_WRA:
            return WRA()
        elif prefix == HEURISTIC_F_MEASURE:
            beta = get_float_argument(args, ARGUMENT_BETA, 0.5, lambda x: x >= 0)
            return FMeasure(beta)
        elif prefix == HEURISTIC_M_ESTIMATE:
            m = get_float_argument(args, ARGUMENT_M, 22.466, lambda x: x >= 0)
            return MEstimate(m)
        raise ValueError('Invalid value given for parameter \'heuristic\': ' + str(heuristic))

    def __create_statistics_provider_factory(self, heuristic: Heuristic) -> StatisticsProviderFactory:
        loss = self.loss

        if loss == AVERAGING_LABEL_WISE:
            default_rule_evaluation_factory = HeuristicLabelWiseRuleEvaluationFactory(heuristic, predictMajority=True)
            rule_evaluation_factory = HeuristicLabelWiseRuleEvaluationFactory(heuristic)
            return LabelWiseStatisticsProviderFactory(default_rule_evaluation_factory, rule_evaluation_factory)
        raise ValueError('Invalid value given for parameter \'loss\': ' + str(loss))

    def __create_lift_function(self, num_labels: int) -> LiftFunction:
        lift_function = self.lift_function

        prefix, args = parse_prefix_and_dict(lift_function, [LIFT_FUNCTION_PEAK])

        if prefix == LIFT_FUNCTION_PEAK:
            peak_label = get_int_argument(args, ARGUMENT_PEAK_LABEL, int(num_labels / 2) + 1,
                                          lambda x: 1 <= x <= num_labels)
            max_lift = get_float_argument(args, ARGUMENT_MAX_LIFT, 1.5, lambda x: x >= 1)
            curvature = get_float_argument(args, ARGUMENT_CURVATURE, 1.0, lambda x: x > 0)
            return PeakLiftFunction(num_labels, peak_label, max_lift, curvature)

        raise ValueError('Invalid value given for parameter \'lift_function\': ' + str(lift_function))

    def __create_head_refinement_factory(self, lift_function: LiftFunction) -> HeadRefinementFactory:
        head_refinement = self.head_refinement

        if head_refinement is None:
            return SingleLabelHeadRefinementFactory()
        elif head_refinement == HEAD_REFINEMENT_SINGLE:
            return SingleLabelHeadRefinementFactory()
        elif head_refinement == HEAD_REFINEMENT_PARTIAL:
            return PartialHeadRefinementFactory(lift_function)
        raise ValueError('Invalid value given for parameter \'head_refinement\': ' + str(head_refinement))

    def _create_predictor(self, num_labels: int, label_matrix: CContiguousLabelMatrix) -> Predictor:
        return self.__create_label_wise_predictor(num_labels)

    def _create_predictor_lil(self, num_labels: int, label_matrix: list) -> Predictor:
        return self.__create_label_wise_predictor(num_labels)

    def __create_label_wise_predictor(self, num_labels: int) -> LabelWiseClassificationPredictor:
        num_threads = get_preferred_num_threads(self.num_threads_prediction)
        return LabelWiseClassificationPredictor(num_labels, num_threads)
