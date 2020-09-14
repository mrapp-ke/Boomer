#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides a scikit-learn implementations of boosting algorithms
"""
from boomer.boosting.example_wise_losses import ExampleWiseLogisticLoss
from boomer.boosting.example_wise_rule_evaluation import RegularizedExampleWiseRuleEvaluation
from boomer.boosting.example_wise_statistics import ExampleWiseStatisticsProviderFactory
from boomer.boosting.label_wise_losses import LabelWiseLoss, LabelWiseLogisticLoss, LabelWiseSquaredErrorLoss
from boomer.boosting.label_wise_rule_evaluation import RegularizedLabelWiseRuleEvaluation
from boomer.boosting.label_wise_statistics import LabelWiseStatisticsProviderFactory
from boomer.boosting.shrinkage import ConstantShrinkage, Shrinkage
from boomer.common.head_refinement import HeadRefinement, SingleLabelHeadRefinement, FullHeadRefinement
from boomer.common.prediction import Predictor, DensePredictor, SignFunction
from boomer.common.rule_induction import ExactGreedyRuleInduction
from boomer.common.rules import ModelBuilder, RuleListBuilder
from boomer.common.sequential_rule_induction import SequentialRuleInduction
from boomer.common.statistics import StatisticsProviderFactory

from boomer.common.rule_learners import INSTANCE_SUB_SAMPLING_BAGGING, FEATURE_SUB_SAMPLING_RANDOM, \
    HEAD_REFINEMENT_SINGLE
from boomer.common.rule_learners import MLRuleLearner, SparsePolicy
from boomer.common.rule_learners import create_pruning, create_feature_sub_sampling, create_instance_sub_sampling, \
    create_label_sub_sampling, create_max_conditions, create_stopping_criteria, create_min_coverage, \
    create_max_head_refinements, create_num_threads

HEAD_REFINEMENT_FULL = 'full'

LOSS_LABEL_WISE_LOGISTIC = 'label-wise-logistic-loss'

LOSS_LABEL_WISE_SQUARED_ERROR = 'label-wise-squared-error-loss'

LOSS_EXAMPLE_WISE_LOGISTIC = 'example-wise-logistic-loss'


class Boomer(MLRuleLearner):
    """
    A scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
    classification rules.
    """

    def __init__(self, random_state: int = 1, feature_format: str = SparsePolicy.AUTO.value,
                 label_format: str = SparsePolicy.AUTO.value, max_rules: int = 1000, time_limit: int = -1,
                 head_refinement: str = None, loss: str = LOSS_LABEL_WISE_LOGISTIC, label_sub_sampling: str = None,
                 instance_sub_sampling: str = INSTANCE_SUB_SAMPLING_BAGGING,
                 feature_sub_sampling: str = FEATURE_SUB_SAMPLING_RANDOM, pruning: str = None, shrinkage: float = 0.3,
                 l2_regularization_weight: float = 1.0, min_coverage: int = 1, max_conditions: int = -1,
                 max_head_refinements: int = 1, num_threads: int = -1):
        """
        :param max_rules:                           The maximum number of rules to be induced (including the default
                                                    rule)
        :param time_limit:                          The duration in seconds after which the induction of rules should be
                                                    canceled
        :param head_refinement:                     The strategy that is used to find the heads of rules. Must be
                                                    `single-label`, `full` or None, if the default strategy should be
                                                    used
        :param loss:                                The loss function to be minimized. Must be
                                                    `label-wise-squared-error-loss`, `label-wise-logistic-loss` or
                                                    `example-wise-logistic-loss`
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
        :param num_threads:                         The number of threads to be used for training or -1, if the number
                                                    of cores available on the machine should be used
        """
        super().__init__(random_state, feature_format, label_format)
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.head_refinement = head_refinement
        self.loss = loss
        self.label_sub_sampling = label_sub_sampling
        self.instance_sub_sampling = instance_sub_sampling
        self.feature_sub_sampling = feature_sub_sampling
        self.pruning = pruning
        self.shrinkage = shrinkage
        self.l2_regularization_weight = l2_regularization_weight
        self.min_coverage = min_coverage
        self.max_conditions = max_conditions
        self.max_head_refinements = max_head_refinements
        self.num_threads = num_threads

    def get_name(self) -> str:
        name = 'max-rules=' + str(self.max_rules)
        if self.head_refinement is not None:
            name += '_head-refinement=' + str(self.head_refinement)
        name += '_loss=' + str(self.loss)
        if self.label_sub_sampling is not None:
            name += '_label-sub-sampling=' + str(self.label_sub_sampling)
        if self.instance_sub_sampling is not None:
            name += '_instance-sub-sampling=' + str(self.instance_sub_sampling)
        if self.feature_sub_sampling is not None:
            name += '_feature-sub-sampling=' + str(self.feature_sub_sampling)
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
        if int(self.max_head_refinements) != 1:
            name += '_max-head-refinements=' + str(self.max_head_refinements)
        if int(self.random_state) != 1:
            name += '_random_state=' + str(self.random_state)
        return name

    def _create_predictor(self) -> Predictor:
        return DensePredictor(SignFunction())

    def _create_model_builder(self) -> ModelBuilder:
        return RuleListBuilder()

    def _create_sequential_rule_induction(self, num_labels: int) -> SequentialRuleInduction:
        stopping_criteria = create_stopping_criteria(int(self.max_rules), int(self.time_limit))
        label_sub_sampling = create_label_sub_sampling(self.label_sub_sampling, num_labels)
        instance_sub_sampling = create_instance_sub_sampling(self.instance_sub_sampling)
        feature_sub_sampling = create_feature_sub_sampling(self.feature_sub_sampling)
        pruning = create_pruning(self.pruning)
        shrinkage = self.__create_shrinkage()
        min_coverage = create_min_coverage(self.min_coverage)
        max_conditions = create_max_conditions(self.max_conditions)
        max_head_refinements = create_max_head_refinements(self.max_head_refinements)
        loss_function = self.__create_loss_function()
        default_rule_head_refinement = FullHeadRefinement()
        head_refinement = self.__create_head_refinement(loss_function)
        l2_regularization_weight = self.__create_l2_regularization_weight()
        rule_evaluation = self.__create_rule_evaluation(loss_function, l2_regularization_weight)
        num_threads = create_num_threads(self.num_threads)
        statistics_provider_factory = self.__create_statistics_provider_factory(loss_function, rule_evaluation)
        rule_induction = ExactGreedyRuleInduction()
        return SequentialRuleInduction(statistics_provider_factory, rule_induction, default_rule_head_refinement,
                                       head_refinement, stopping_criteria, label_sub_sampling, instance_sub_sampling,
                                       feature_sub_sampling, pruning, shrinkage, min_coverage, max_conditions,
                                       max_head_refinements, num_threads)

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
        elif loss == LOSS_LABEL_WISE_LOGISTIC:
            return LabelWiseLogisticLoss()
        elif loss == LOSS_EXAMPLE_WISE_LOGISTIC:
            return ExampleWiseLogisticLoss()
        raise ValueError('Invalid value given for parameter \'loss\': ' + str(loss))

    def __create_rule_evaluation(self, loss_function, l2_regularization_weight: float):
        if isinstance(loss_function, LabelWiseLoss):
            return RegularizedLabelWiseRuleEvaluation(l2_regularization_weight)
        else:
            return RegularizedExampleWiseRuleEvaluation(l2_regularization_weight)

    def __create_statistics_provider_factory(self, loss_function, rule_evaluation) -> StatisticsProviderFactory:
        if isinstance(loss_function, LabelWiseLoss):
            return LabelWiseStatisticsProviderFactory(loss_function, rule_evaluation, rule_evaluation)
        else:
            return ExampleWiseStatisticsProviderFactory(loss_function, rule_evaluation, rule_evaluation)

    def __create_head_refinement(self, loss_function) -> HeadRefinement:
        head_refinement = self.head_refinement

        if head_refinement is None:
            if isinstance(loss_function, LabelWiseLoss):
                return SingleLabelHeadRefinement()
            else:
                return FullHeadRefinement()
        elif head_refinement == HEAD_REFINEMENT_SINGLE:
            return SingleLabelHeadRefinement()
        elif head_refinement == HEAD_REFINEMENT_FULL:
            return FullHeadRefinement()
        raise ValueError('Invalid value given for parameter \'head_refinement\': ' + str(head_refinement))

    def __create_shrinkage(self) -> Shrinkage:
        shrinkage = float(self.shrinkage)

        if 0.0 < shrinkage < 1.0:
            return ConstantShrinkage(shrinkage)
        if shrinkage == 1.0:
            return None
        raise ValueError('Invalid value given for parameter \'shrinkage\': ' + str(shrinkage))
