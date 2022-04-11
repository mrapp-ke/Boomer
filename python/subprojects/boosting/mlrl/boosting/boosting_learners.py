#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides scikit-learn implementations of boosting algorithms.
"""
from typing import Dict, Set, Optional

from mlrl.boosting.cython.learner import BoostingRuleLearner as BoostingRuleLearnerWrapper, BoostingRuleLearnerConfig
from mlrl.common.cython.learner import RuleLearnerConfig, RuleLearner as RuleLearnerWrapper
from mlrl.common.cython.stopping_criterion import AggregationFunction
from mlrl.common.options import BooleanOption
from mlrl.common.rule_learners import AUTOMATIC, NONE, ARGUMENT_BIN_RATIO, \
    ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS, ARGUMENT_NUM_THREADS, BINNING_EQUAL_WIDTH, BINNING_EQUAL_FREQUENCY
from mlrl.common.rule_learners import MLRuleLearner, SparsePolicy
from mlrl.common.rule_learners import configure_rule_model_assemblage, configure_rule_induction, \
    configure_feature_binning, configure_label_sampling, configure_instance_sampling, configure_feature_sampling, \
    configure_partition_sampling, configure_pruning, configure_parallel_rule_refinement, \
    configure_parallel_statistic_update, configure_parallel_prediction, configure_size_stopping_criterion, \
    configure_time_stopping_criterion
from mlrl.common.rule_learners import parse_param, parse_param_and_options
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

HEAD_TYPE_SINGLE = 'single-label'

HEAD_TYPE_COMPLETE = 'complete'

LOSS_LOGISTIC_LABEL_WISE = 'logistic-label-wise'

LOSS_LOGISTIC_EXAMPLE_WISE = 'logistic-example-wise'

LOSS_SQUARED_ERROR_LABEL_WISE = 'squared-error-label-wise'

LOSS_SQUARED_HINGE_LABEL_WISE = 'squared-hinge-label-wise'

NON_DECOMPOSABLE_LOSSES = {LOSS_LOGISTIC_EXAMPLE_WISE}

PREDICTOR_LABEL_WISE = 'label-wise'

PREDICTOR_EXAMPLE_WISE = 'example-wise'

HEAD_TYPE_VALUES: Set[str] = {
    HEAD_TYPE_SINGLE,
    HEAD_TYPE_COMPLETE,
    AUTOMATIC
}

EARLY_STOPPING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    EARLY_STOPPING_LOSS: {ARGUMENT_AGGREGATION_FUNCTION, ARGUMENT_MIN_RULES, ARGUMENT_UPDATE_INTERVAL,
                          ARGUMENT_STOP_INTERVAL, ARGUMENT_NUM_PAST, ARGUMENT_NUM_RECENT, ARGUMENT_MIN_IMPROVEMENT,
                          ARGUMENT_FORCE_STOP}
}

FEATURE_BINNING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    BINNING_EQUAL_FREQUENCY: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS},
    BINNING_EQUAL_WIDTH: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS},
    AUTOMATIC: {},
}

LABEL_BINNING_VALUES: Dict[str, Set[str]] = {
    NONE: {},
    BINNING_EQUAL_WIDTH: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS},
    AUTOMATIC: {}
}

LOSS_VALUES: Set[str] = {
    LOSS_SQUARED_ERROR_LABEL_WISE,
    LOSS_SQUARED_HINGE_LABEL_WISE,
    LOSS_LOGISTIC_LABEL_WISE,
    LOSS_LOGISTIC_EXAMPLE_WISE
}

PREDICTOR_VALUES: Set[str] = {
    PREDICTOR_LABEL_WISE,
    PREDICTOR_EXAMPLE_WISE,
    AUTOMATIC
}

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

    def __init__(self,
                 random_state: int = 1,
                 feature_format: str = SparsePolicy.AUTO.value,
                 label_format: str = SparsePolicy.AUTO.value,
                 prediction_format: str = SparsePolicy.AUTO.value,
                 rule_model_assemblage: Optional[str] = None,
                 rule_induction: Optional[str] = None,
                 max_rules: Optional[int] = None,
                 time_limit: Optional[int] = None,
                 early_stopping: Optional[str] = None,
                 head_type: Optional[str] = None,
                 loss: Optional[str] = None,
                 predictor: Optional[str] = AUTOMATIC,
                 label_sampling: Optional[str] = None,
                 instance_sampling: Optional[str] = None,
                 feature_sampling: Optional[str] = None,
                 holdout: Optional[str] = None,
                 feature_binning: Optional[str] = None,
                 label_binning: Optional[str] = None,
                 pruning: Optional[str] = None,
                 shrinkage: Optional[float] = 0.3,
                 l1_regularization_weight: Optional[float] = None,
                 l2_regularization_weight: Optional[float] = None,
                 parallel_rule_refinement: Optional[str] = None,
                 parallel_statistic_update: Optional[str] = None,
                 parallel_prediction: Optional[str] = None):
        """
        :param rule_model_assemblage:       The algorithm that should be used for the induction of several rules. Must
                                            be 'sequential'. For additional options refer to the documentation
        :param rule_induction:              The algorithm that should be used for the induction of individual rules.
                                            Must be 'top-down'. For additional options refer to the documentation
        :param max_rules:                   The maximum number of rules to be learned (including the default rule). Must
                                            be at least 1 or 0, if the number of rules should not be restricted.
        :param time_limit:                  The duration in seconds after which the induction of rules should be
                                            canceled. Must be at least 1 or 0, if no time limit should be set
        :param early_stopping:              The strategy that should be used for early stopping. Must be 'loss', if the
                                            induction of new rules should be stopped as soon as the performance of the
                                            model does not improve on a holdout set according to the loss function or
                                            'none', if no early stopping should be used. For additional options refer to
                                            the documentation
        :param head_type:                   The type of the rule heads that should be used. Must be 'single-label',
                                            'complete' or 'auto', if the type of the heads should be chosen
                                            automatically
        :param loss:                        The loss function to be minimized. Must be 'squared-error-label-wise',
                                            'squared-hinge-label-wise', 'logistic-label-wise' or 'logistic-example-wise'
        :param predictor:                   The strategy that should be used for making predictions. Must be
                                            'label-wise', 'example-wise' or 'auto', if the most suitable strategy should
                                            be chosen automatically, depending on the loss function
        :param label_sampling:              The strategy that should be used to sample from the available labels
                                            whenever a new rule is learned. Must be 'without-replacement' or 'none', if
                                            no sampling should be used. For additional options refer to the
                                            documentation
        :param instance_sampling:           The strategy that should be used to sample from the available the training
                                            examples whenever a new rule is learned. Must be 'with-replacement',
                                            'without-replacement', 'stratified_label_wise', 'stratified_example_wise' or
                                            'none', if no sampling should be used. For additional options refer to the
                                            documentation
        :param feature_sampling:            The strategy that is used to sample from the available features whenever a
                                            rule is refined. Must be 'without-replacement' or 'none', if no sampling
                                            should be used. For additional options refer to the documentation
        :param holdout:                     The name of the strategy that should be used to creating a holdout set. Must
                                            be 'random', 'stratified-label-wise', 'stratified-example-wise' or 'none',
                                            if no holdout set should be used. For additional options refer to the
                                            documentation
        :param feature_binning:             The strategy that should be used to assign examples to bins based on their
                                            feature values. Must be 'auto', 'equal-width', 'equal-frequency' or 'none',
                                            if no feature binning should be used. If set to 'auto', the most suitable
                                            strategy is chosen automatically, depending on the characteristics of the
                                            feature matrix. For additional options refer to the documentation
        :param label_binning:               The strategy that should be used to assign labels to bins. Must be 'auto',
                                            'equal-width' or 'none', if no label binning should be used. If set to
                                            'auto', the most suitable strategy is chosen automatically, depending on the
                                            loss function and the type of rule heads. For additional options refer to
                                            the documentation
        :param pruning:                     The strategy that should be used to prune individual rules. Must be 'irep'
                                            or 'none', if no pruning should be used
        :param shrinkage:                   The shrinkage parameter, a.k.a. the "learning rate", that should be used to
                                            shrink the weight of individual rules. Must be in (0, 1]
        :param l1_regularization_weight:    The weight of the L1 regularization. Must be at least 0
        :param l2_regularization_weight:    The weight of the L2 regularization. Must be at least 0
        :param parallel_rule_refinement:    Whether potential refinements of rules should be searched for in parallel or
                                            not. Must be 'true', 'false' or 'auto', if the most suitable strategy should
                                            be chosen automatically depending on the loss function. For additional
                                            options refer to the documentation
        :param parallel_statistic_update:   Whether the gradients and Hessians for different examples should be updated
                                            in parallel or not. Must be 'true', 'false' or 'auto', if the most suitable
                                            strategy should be chosen automatically, depending on the loss function. For
                                            additional options refer to the documentation
        :param parallel_prediction:         Whether predictions for different examples should be obtained in parallel or
                                            not. Must be 'true' or 'false'. For additional options refer to the
                                            documentation
        """
        super().__init__(random_state, feature_format, label_format, prediction_format)
        self.rule_model_assemblage = rule_model_assemblage
        self.rule_induction = rule_induction
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.early_stopping = early_stopping
        self.head_type = head_type
        self.loss = loss
        self.predictor = predictor
        self.label_sampling = label_sampling
        self.instance_sampling = instance_sampling
        self.feature_sampling = feature_sampling
        self.holdout = holdout
        self.feature_binning = feature_binning
        self.label_binning = label_binning
        self.pruning = pruning
        self.shrinkage = shrinkage
        self.l1_regularization_weight = l1_regularization_weight
        self.l2_regularization_weight = l2_regularization_weight
        self.parallel_rule_refinement = parallel_rule_refinement
        self.parallel_statistic_update = parallel_statistic_update
        self.parallel_prediction = parallel_prediction

    def get_name(self) -> str:
        name = 'boomer'
        if self.random_state != 1:
            name += '_random-state=' + str(self.random_state)
        if self.feature_format != SparsePolicy.AUTO.value:
            name += '_feature-format=' + str(self.feature_format)
        if self.label_format != SparsePolicy.AUTO.value:
            name += '_label-format=' + str(self.label_format)
        if self.prediction_format != SparsePolicy.AUTO.value:
            name += '_prediction-format=' + str(self.prediction_format)
        if self.rule_model_assemblage is not None:
            name += '_rule-model-assemblage=' + str(self.rule_model_assemblage)
        if self.rule_induction is not None:
            name += '_rule-induction=' + str(self.rule_induction)
        if self.max_rules is not None:
            name += '_max-rules=' + str(self.max_rules)
        if self.time_limit is not None:
            name += '_time-limit=' + str(self.time_limit)
        if self.early_stopping is not None:
            name += '_early-stopping=' + str(self.early_stopping)
        if self.head_type is not None:
            name += '_head-type=' + str(self.head_type)
        if self.loss is not None:
            name += '_loss=' + str(self.loss)
        if self.predictor is not None:
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
        if self.label_binning is not None:
            name += '_label-binning=' + str(self.label_binning)
        if self.pruning is not None:
            name += '_pruning=' + str(self.pruning)
        if self.shrinkage is not None:
            name += '_shrinkage=' + str(self.shrinkage)
        if self.l1_regularization_weight is not None:
            name += '_l1=' + str(self.l1_regularization_weight)
        if self.l2_regularization_weight is not None:
            name += '_l2=' + str(self.l2_regularization_weight)
        if self.parallel_rule_refinement is not None:
            name += '_parallel-rule-refinement=' + str(self.parallel_rule_refinement)
        if self.parallel_statistic_update is not None:
            name += '_parallel-statistic-update=' + str(self.parallel_statistic_update)
        if self.parallel_prediction is not None:
            name += '_parallel-prediction=' + str(self.parallel_prediction)
        return name

    def _create_learner(self) -> RuleLearnerWrapper:
        config = BoostingRuleLearnerConfig()
        configure_rule_model_assemblage(config, self.rule_model_assemblage)
        configure_rule_induction(config, self.rule_induction)
        self.__configure_feature_binning(config)
        configure_label_sampling(config, self.label_sampling)
        configure_instance_sampling(config, self.instance_sampling)
        configure_feature_sampling(config, self.feature_sampling)
        configure_partition_sampling(config, self.holdout)
        configure_pruning(config, self.pruning)
        self.__configure_parallel_rule_refinement(config)
        self.__configure_parallel_statistic_update(config)
        configure_parallel_prediction(config, self.parallel_prediction)
        configure_size_stopping_criterion(config, max_rules=self.max_rules)
        configure_time_stopping_criterion(config, time_limit=self.time_limit)
        self.__configure_measure_stopping_criterion(config)
        self.__configure_post_processor(config)
        self.__configure_head_type(config)
        self.__configure_l1_regularization(config)
        self.__configure_l2_regularization(config)
        self.__configure_loss(config)
        self.__configure_label_binning(config)
        self.__configure_classification_predictor(config)
        return BoostingRuleLearnerWrapper(config)

    def __configure_feature_binning(self, config: BoostingRuleLearnerConfig):
        feature_binning = self.feature_binning

        if feature_binning is not None:
            if feature_binning == AUTOMATIC:
                config.use_automatic_feature_binning()
            else:
                configure_feature_binning(config, feature_binning)

    def __configure_parallel_rule_refinement(self, config: BoostingRuleLearnerConfig):
        parallel_rule_refinement = self.parallel_rule_refinement

        if parallel_rule_refinement is not None:
            if parallel_rule_refinement == AUTOMATIC:
                config.use_automatic_parallel_rule_refinement()
            else:
                configure_parallel_rule_refinement(config, parallel_rule_refinement)

    def __configure_parallel_statistic_update(self, config: BoostingRuleLearnerConfig):
        parallel_statistic_update = self.parallel_statistic_update

        if parallel_statistic_update is not None:
            if parallel_statistic_update == AUTOMATIC:
                config.use_automatic_parallel_statistic_update()
            else:
                configure_parallel_statistic_update(config, parallel_statistic_update)

    def __configure_measure_stopping_criterion(self, config: RuleLearnerConfig):
        early_stopping = self.early_stopping

        if early_stopping is not None:
            value, options = parse_param_and_options('early_stopping', early_stopping, EARLY_STOPPING_VALUES)

            if value == NONE:
                config.use_no_measure_stopping_criterion()
            elif value == EARLY_STOPPING_LOSS:
                c = config.use_measure_stopping_criterion()
                aggregation_function = options.get_string(ARGUMENT_AGGREGATION_FUNCTION, None)
                c.set_aggregation_function(self.__create_aggregation_function(
                    aggregation_function) if aggregation_function is not None else c.get_aggregation_function())
                c.set_min_rules(options.get_int(ARGUMENT_MIN_RULES, c.get_min_rules()))
                c.set_update_interval(options.get_int(ARGUMENT_UPDATE_INTERVAL, c.get_update_interval()))
                c.set_stop_interval(options.get_int(ARGUMENT_STOP_INTERVAL, c.get_stop_interval()))
                c.set_num_past(options.get_int(ARGUMENT_NUM_PAST, c.get_num_past()))
                c.set_num_current(options.get_int(ARGUMENT_NUM_RECENT, c.get_num_current()))
                c.set_min_improvement(options.get_float(ARGUMENT_MIN_IMPROVEMENT, c.get_min_improvement()))
                c.set_force_stop(options.get_bool(ARGUMENT_FORCE_STOP, c.get_force_stop()))

    @staticmethod
    def __create_aggregation_function(aggregation_function: str) -> AggregationFunction:
        value = parse_param(ARGUMENT_AGGREGATION_FUNCTION, aggregation_function, {AGGREGATION_FUNCTION_MIN,
                                                                                  AGGREGATION_FUNCTION_MAX,
                                                                                  AGGREGATION_FUNCTION_ARITHMETIC_MEAN})

        if value == AGGREGATION_FUNCTION_MIN:
            return AggregationFunction.MIN
        elif value == AGGREGATION_FUNCTION_MAX:
            return AggregationFunction.MAX
        elif value == AGGREGATION_FUNCTION_ARITHMETIC_MEAN:
            return AggregationFunction.ARITHMETIC_MEAN

    def __configure_post_processor(self, config: BoostingRuleLearnerConfig):
        shrinkage = self.shrinkage

        if shrinkage is not None:
            if shrinkage == 1:
                config.use_no_post_processor()
            else:
                config.use_constant_shrinkage_post_processor().set_shrinkage(shrinkage)

    def __configure_head_type(self, config: BoostingRuleLearnerConfig):
        head_type = self.head_type

        if head_type is not None:
            value = parse_param("head_type", head_type, HEAD_TYPE_VALUES)

            if value == AUTOMATIC:
                config.use_automatic_heads()
            elif value == HEAD_TYPE_SINGLE:
                config.use_single_label_heads()
            elif value == HEAD_TYPE_COMPLETE:
                config.use_complete_heads()

    def __configure_l1_regularization(self, config: BoostingRuleLearnerConfig):
        l1_regularization_weight = self.l1_regularization_weight

        if l1_regularization_weight is not None:
            if l1_regularization_weight == 0:
                config.use_no_l1_regularization()
            else:
                config.use_l1_regularization().set_regularization_weight(l1_regularization_weight)

    def __configure_l2_regularization(self, config: BoostingRuleLearnerConfig):
        l2_regularization_weight = self.l2_regularization_weight

        if l2_regularization_weight is not None:
            if l2_regularization_weight == 0:
                config.use_no_l2_regularization()
            else:
                config.use_l2_regularization().set_regularization_weight(l2_regularization_weight)

    def __configure_loss(self, config: BoostingRuleLearnerConfig):
        loss = self.loss

        if loss is not None:
            value = parse_param("loss", loss, LOSS_VALUES)

            if value == LOSS_SQUARED_ERROR_LABEL_WISE:
                config.use_label_wise_squared_error_loss()
            elif value == LOSS_SQUARED_HINGE_LABEL_WISE:
                config.use_label_wise_squared_hinge_loss()
            elif value == LOSS_LOGISTIC_LABEL_WISE:
                config.use_label_wise_logistic_loss()
            elif value == LOSS_LOGISTIC_EXAMPLE_WISE:
                config.use_example_wise_logistic_loss()

    def __configure_label_binning(self, config: BoostingRuleLearnerConfig):
        label_binning = self.label_binning

        if label_binning is not None:
            value, options = parse_param_and_options('label_binning', label_binning, LABEL_BINNING_VALUES)

            if value == NONE:
                config.use_no_label_binning()
            elif value == AUTOMATIC:
                config.use_automatic_label_binning()
            if value == BINNING_EQUAL_WIDTH:
                c = config.use_equal_width_label_binning()
                c.set_bin_ratio(options.get_float(ARGUMENT_BIN_RATIO, c.get_bin_ratio()))
                c.set_min_bins(options.get_int(ARGUMENT_MIN_BINS, c.get_min_bins()))
                c.set_max_bins(options.get_int(ARGUMENT_MAX_BINS, c.get_max_bins()))

    def __configure_classification_predictor(self, config: BoostingRuleLearnerConfig):
        predictor = self.predictor

        if predictor is not None:
            value = parse_param('predictor', predictor, PREDICTOR_VALUES)

            if predictor == AUTOMATIC:
                config.use_automatic_label_binning()
            elif value == PREDICTOR_LABEL_WISE:
                config.use_label_wise_classification_predictor()
            elif value == PREDICTOR_EXAMPLE_WISE:
                config.use_example_wise_classification_predictor()
