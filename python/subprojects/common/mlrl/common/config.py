"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities that ease the configuration of rule learning algorithms.
"""
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Dict, Optional, Set

from mlrl.common.cython.learner import BeamSearchTopDownRuleInductionMixin, EqualFrequencyFeatureBinningMixin, \
    EqualWidthFeatureBinningMixin, ExampleWiseStratifiedBiPartitionSamplingMixin, \
    ExampleWiseStratifiedInstanceSamplingMixin, FeatureSamplingWithoutReplacementMixin, \
    GreedyTopDownRuleInductionMixin, InstanceSamplingWithoutReplacementMixin, InstanceSamplingWithReplacementMixin, \
    IrepRulePruningMixin, LabelWiseStratifiedBiPartitionSamplingMixin, LabelWiseStratifiedInstanceSamplingMixin, \
    NoFeatureBinningMixin, NoFeatureSamplingMixin, NoGlobalPruningMixin, NoInstanceSamplingMixin, \
    NoParallelPredictionMixin, NoParallelRuleRefinementMixin, NoParallelStatisticUpdateMixin, \
    NoPartitionSamplingMixin, NoRulePruningMixin, NoSequentialPostOptimizationMixin, NoSizeStoppingCriterionMixin, \
    NoTimeStoppingCriterionMixin, ParallelPredictionMixin, ParallelRuleRefinementMixin, ParallelStatisticUpdateMixin, \
    PostPruningMixin, PrePruningMixin, RandomBiPartitionSamplingMixin, RoundRobinLabelSamplingMixin, \
    SequentialPostOptimizationMixin, SizeStoppingCriterionMixin, TimeStoppingCriterionMixin
from mlrl.common.cython.stopping_criterion import AggregationFunction
from mlrl.common.format import format_dict_keys, format_string_set
from mlrl.common.options import BooleanOption, Options, parse_param, parse_param_and_options

AUTOMATIC = 'auto'

NONE = 'none'

OPTION_RESAMPLE_FEATURES = 'resample_features'

SAMPLING_WITH_REPLACEMENT = 'with-replacement'

SAMPLING_WITHOUT_REPLACEMENT = 'without-replacement'

SAMPLING_STRATIFIED_LABEL_WISE = 'stratified-label-wise'

SAMPLING_STRATIFIED_EXAMPLE_WISE = 'stratified-example-wise'

OPTION_SAMPLE_SIZE = 'sample_size'

OPTION_NUM_SAMPLES = 'num_samples'

BINNING_EQUAL_FREQUENCY = 'equal-frequency'

BINNING_EQUAL_WIDTH = 'equal-width'

OPTION_BIN_RATIO = 'bin_ratio'

OPTION_MIN_BINS = 'min_bins'

OPTION_MAX_BINS = 'max_bins'

OPTION_NUM_THREADS = 'num_threads'

OPTION_USE_HOLDOUT_SET = 'use_holdout_set'


class Parameter(ABC):
    """
    An abstract base class for all parameters of a rule learning algorithm.
    """

    def __init__(self, name: str, description: str):
        """
        :param name:        The name of the parameter
        :param description: A textual description of the parameter
        """
        self.name = name
        self.description = description

    @abstractmethod
    def configure(self, config, value):
        """
        Configures a rule learner depending on this parameter.

        :param config:  The configuration to be modified
        :param value:   The value to be set
        """
        pass

    @abstractmethod
    def add_to_argument_parser(self, parser: ArgumentParser, config_type: type):
        """
        Adds a command line argument that corresponds to this parameter to an `ArgumentParser` if it is supported by a
        configuration of a specific type.
        """
        pass

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    @property
    def argument_name(self) -> str:
        return '--' + self.name.replace('_', '-')


class NominalParameter(Parameter, ABC):
    """
    An abstract base class for all nominal parameters of a rule learning algorithm that allow to set one out of a set of
    predefined values.
    """

    class Value:
        """
        A value that can be set for a nominal parameter.
        """

        def __init__(self, name: str, mixin: type, options: Set[str], description: Optional[str]):
            """
            :param name:        The name of the value
            :param mixin:       The type of the mixin that must be implemented by a rule learner to support this value
            :param options:     A set that contains the names of additional options that may be specified
            :param description: A textual description of the value
            """
            self.name = name
            self.mixin = mixin
            self.options = options
            self.description = description

        def __eq__(self, other):
            return self.name == other.name

        def __hash__(self):
            return hash(self.name)

    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.values = {}

    def add_value(self, name: str, mixin: type, options: Set[str] = set(), description: Optional[str] = None):
        """
        Adds a new value to the parameter.

        :param name:        The name of the value to be added
        :param mixin:       The type of the mixin that must be implemented by a rule learner to support the value
        :param options:     A set that contains the names of additional options that may be specified
        :param description: A textual description of the value
        :return:            The parameter itself
        """
        self.values[name] = NominalParameter.Value(name=name, mixin=mixin, options=options, description=description)
        return self

    @abstractmethod
    def _configure(self, config, value: str, options: Optional[Options]):
        """
        Must be implemented by subclasses in order to configure a rule learner depending on the specified nominal value.
        
        :param config:  The configuration to be modified
        :param value:   The nominal value to be set
        :param options: Additional options that have eventually been specified
        """
        pass

    def __get_supported_values(self, config_type: type):
        num_options = 0
        supported_values = {}

        for parameter_value in self.values.values():
            if issubclass(config_type, parameter_value.mixin):
                options = parameter_value.options
                num_options += len(options)
                supported_values[parameter_value.name] = options

        if num_options == 0:
            supported_values = {value for value in supported_values.keys()}

        return supported_values

    def configure(self, config, value):
        supported_values = self.__get_supported_values(type(config))

        if len(supported_values) > 0:
            value = str(value)

            if type(supported_values) == dict:
                value, options = parse_param_and_options(self.name, value, supported_values)
            else:
                value = parse_param(self.name, value, supported_values)
                options = None

            self._configure(config, value, options)

    def add_to_argument_parser(self, parser: ArgumentParser, config_type: type):
        supported_values = self.__get_supported_values(config_type)

        if len(supported_values) > 0:
            description = self.description
            description += '. Must be one of '

            if type(supported_values) == dict:
                description += format_dict_keys(supported_values)
                suffix = ' For additional options refer to the documentation.'
                supported_values = {value for value in supported_values.keys()}
            else:
                description += format_string_set(supported_values)
                suffix = ''

            description += '.'

            for supported_value in supported_values:
                value_description = self.values[supported_value].description

                if value_description is not None and len(value_description) > 0:
                    description += ' ' + value_description + '.'

            description += suffix
            parser.add_argument(self.argument_name, type=str, help=description)


class NumericalParameter(Parameter, ABC):
    """
    An abstract base class for all parameters of a rule learning algorithm that allow to set a numerical value.
    """

    def __init__(self, name: str, description: str, mixin: type, numeric_type: type):
        """
        :param mixin:           The type of the mixin that must be implemented by a rule learner to support the
                                parameter
        :param numeric_type:    The type of the numerical value
        """
        super().__init__(name=name, description=description)
        self.mixin = mixin
        self.numeric_type = numeric_type

    @abstractmethod
    def _configure(self, config, value):
        """
        Must be implemented by subclasses in order to configure a rule learner depending on the specified numerical
        value.
        
        :param config:  The configuration to be modified
        :param value:   The numerical value to be set
        """
        pass

    def __is_supported(self, config_type: type):
        return issubclass(config_type, self.mixin)

    def configure(self, config, value):
        if self.__is_supported(type(config)):
            self._configure(config, self.numeric_type(value))

    def add_to_argument_parser(self, parser: ArgumentParser, config_type: type):
        if self.__is_supported(config_type):
            parser.add_argument(self.argument_name, type=self.numeric_type, help=self.description)


class IntParameter(NumericalParameter, ABC):
    """
    An abstract base class for all parameters of a rule learning algorithm that allow to set an integer value.
    """

    def __init__(self, name: str, description: str, mixin: type):
        """
        :param mixin: The type of the mixin that must be implemented by a rule learner to support the parameter
        """
        super().__init__(name=name, description=description, mixin=mixin, numeric_type=int)


class FloatParameter(NumericalParameter, ABC):
    """
    An abstract base class for all parameters of a rule learning algorithm that allow to set a floating point value.
    """

    def __init__(self, name: str, description: str, mixin: type):
        super().__init__(name=name, description=description, mixin=mixin, numeric_type=float)

    def _cast_value(self, value):
        return float(value)


class RuleInductionParameter(NominalParameter):
    """
    A parameter that allows to configure the algorithm to be used for the induction of individual rules.
    """

    RULE_INDUCTION_TOP_DOWN_GREEDY = 'top-down-greedy'

    OPTION_MIN_COVERAGE = 'min_coverage'

    OPTION_MIN_SUPPORT = 'min_support'

    OPTION_MAX_CONDITIONS = 'max_conditions'

    OPTION_MAX_HEAD_REFINEMENTS = 'max_head_refinements'

    OPTION_RECALCULATE_PREDICTIONS = 'recalculate_predictions'

    RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH = 'top-down-beam-search'

    OPTION_BEAM_WIDTH = 'beam_width'

    def __init__(self):
        super().__init__(name='rule_induction',
                         description='The name of the algorithm to be used for the induction of individual rules')
        self.add_value(name=self.RULE_INDUCTION_TOP_DOWN_GREEDY,
                       mixin=GreedyTopDownRuleInductionMixin,
                       options={
                           self.OPTION_MIN_COVERAGE, self.OPTION_MIN_SUPPORT, self.OPTION_MAX_CONDITIONS,
                           self.OPTION_MAX_HEAD_REFINEMENTS, self.OPTION_RECALCULATE_PREDICTIONS
                       })
        self.add_value(name=self.RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH,
                       mixin=BeamSearchTopDownRuleInductionMixin,
                       options={
                           self.OPTION_MIN_COVERAGE, self.OPTION_MIN_SUPPORT, self.OPTION_MAX_CONDITIONS,
                           self.OPTION_MAX_HEAD_REFINEMENTS, self.OPTION_RECALCULATE_PREDICTIONS,
                           self.OPTION_BEAM_WIDTH, OPTION_RESAMPLE_FEATURES
                       })

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == self.RULE_INDUCTION_TOP_DOWN_GREEDY:
            c = config.use_greedy_top_down_rule_induction()
            c.set_min_coverage(options.get_int(self.OPTION_MIN_COVERAGE, c.get_min_coverage()))
            c.set_min_support(options.get_float(self.OPTION_MIN_SUPPORT, c.get_min_support()))
            c.set_max_conditions(options.get_int(self.OPTION_MAX_CONDITIONS, c.get_max_conditions()))
            c.set_max_head_refinements(options.get_int(self.OPTION_MAX_HEAD_REFINEMENTS, c.get_max_head_refinements()))
            c.set_recalculate_predictions(
                options.get_bool(self.OPTION_RECALCULATE_PREDICTIONS, c.are_predictions_recalculated()))
        elif value == self.RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH:
            c = config.use_beam_search_top_down_rule_induction()
            c.set_min_coverage(options.get_int(self.OPTION_MIN_COVERAGE, c.get_min_coverage()))
            c.set_min_support(options.get_float(self.OPTION_MIN_SUPPORT, c.get_min_support()))
            c.set_max_conditions(options.get_int(self.OPTION_MAX_CONDITIONS, c.get_max_conditions()))
            c.set_max_head_refinements(options.get_int(self.OPTION_MAX_HEAD_REFINEMENTS, c.get_max_head_refinements()))
            c.set_recalculate_predictions(
                options.get_bool(self.OPTION_RECALCULATE_PREDICTIONS, c.are_predictions_recalculated()))
            c.set_beam_width(options.get_int(self.OPTION_BEAM_WIDTH, c.get_beam_width()))
            c.set_resample_features(options.get_bool(OPTION_RESAMPLE_FEATURES, c.are_features_resampled()))


class FeatureBinningParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for feature binning.
    """

    def __init__(self):
        super().__init__(name='feature_binning', description='The name of the strategy to be used for feature binning')
        self.add_value(name=NONE, mixin=NoFeatureBinningMixin)
        self.add_value(name=BINNING_EQUAL_FREQUENCY,
                       mixin=EqualFrequencyFeatureBinningMixin,
                       options={OPTION_BIN_RATIO, OPTION_MIN_BINS, OPTION_MAX_BINS})
        self.add_value(name=BINNING_EQUAL_WIDTH,
                       mixin=EqualWidthFeatureBinningMixin,
                       options={OPTION_BIN_RATIO, OPTION_MIN_BINS, OPTION_MAX_BINS})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == NONE:
            config.use_no_feature_binning()
        elif value == BINNING_EQUAL_FREQUENCY:
            c = config.use_equal_frequency_feature_binning()
            c.set_bin_ratio(options.get_float(OPTION_BIN_RATIO, c.get_bin_ratio()))
            c.set_min_bins(options.get_int(OPTION_MIN_BINS, c.get_min_bins()))
            c.set_max_bins(options.get_int(OPTION_MAX_BINS, c.get_max_bins()))
        elif value == BINNING_EQUAL_WIDTH:
            c = config.use_equal_width_feature_binning()
            c.set_bin_ratio(options.get_float(OPTION_BIN_RATIO, c.get_bin_ratio()))
            c.set_min_bins(options.get_int(OPTION_MIN_BINS, c.get_min_bins()))
            c.set_max_bins(options.get_int(OPTION_MAX_BINS, c.get_max_bins()))


class LabelSamplingParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for label sampling.
    """

    LABEL_SAMPLING_ROUND_ROBIN = 'round-robin'

    def __init__(self):
        super().__init__(name='label_sampling', description='The name of the strategy to be used for label sampling')
        self.add_value(name=NONE, mixin=NoFeatureSamplingMixin)
        self.add_value(name=SAMPLING_WITHOUT_REPLACEMENT,
                       mixin=FeatureSamplingWithoutReplacementMixin,
                       options={OPTION_NUM_SAMPLES})
        self.add_value(name=self.LABEL_SAMPLING_ROUND_ROBIN, mixin=RoundRobinLabelSamplingMixin)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == NONE:
            config.use_no_label_sampling()
        elif value == SAMPLING_WITHOUT_REPLACEMENT:
            c = config.use_label_sampling_without_replacement()
            c.set_num_samples(options.get_int(OPTION_NUM_SAMPLES, c.get_num_samples()))
        elif value == self.LABEL_SAMPLING_ROUND_ROBIN:
            config.use_round_robin_label_sampling()


class InstanceSamplingParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for instance sampling.
    """

    def __init__(self):
        super().__init__(name='instance_sampling',
                         description='The name of the strategy to be used for instance sampling')
        self.add_value(name=NONE, mixin=NoInstanceSamplingMixin)
        self.add_value(name=SAMPLING_WITH_REPLACEMENT,
                       mixin=InstanceSamplingWithReplacementMixin,
                       options={OPTION_SAMPLE_SIZE})
        self.add_value(name=SAMPLING_WITHOUT_REPLACEMENT,
                       mixin=InstanceSamplingWithoutReplacementMixin,
                       options={OPTION_SAMPLE_SIZE})
        self.add_value(name=SAMPLING_STRATIFIED_LABEL_WISE,
                       mixin=LabelWiseStratifiedInstanceSamplingMixin,
                       options={OPTION_SAMPLE_SIZE})
        self.add_value(name=SAMPLING_STRATIFIED_EXAMPLE_WISE,
                       mixin=ExampleWiseStratifiedInstanceSamplingMixin,
                       options={OPTION_SAMPLE_SIZE})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == NONE:
            config.use_no_instance_sampling()
        elif value == SAMPLING_WITH_REPLACEMENT:
            c = config.use_instance_sampling_with_replacement()
            c.set_sample_size(options.get_float(OPTION_SAMPLE_SIZE, c.get_sample_size()))
        elif value == SAMPLING_WITHOUT_REPLACEMENT:
            c = config.use_instance_sampling_without_replacement()
            c.set_sample_size(options.get_float(OPTION_SAMPLE_SIZE, c.get_sample_size()))
        elif value == SAMPLING_STRATIFIED_LABEL_WISE:
            c = config.use_label_wise_stratified_instance_sampling()
            c.set_sample_size(options.get_float(OPTION_SAMPLE_SIZE, c.get_sample_size()))
        elif value == SAMPLING_STRATIFIED_EXAMPLE_WISE:
            c = config.use_example_wise_stratified_instance_sampling()
            c.set_sample_size(options.get_float(OPTION_SAMPLE_SIZE, c.get_sample_size()))


class FeatureSamplingParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for feature sampling.
    """

    OPTION_NUM_RETAINED = 'num_retained'

    def __init__(self):
        super().__init__(name='feature_sampling',
                         description='The name of the strategy to be used for feature sampling')
        self.add_value(name=NONE, mixin=NoFeatureSamplingMixin)
        self.add_value(name=SAMPLING_WITHOUT_REPLACEMENT,
                       mixin=FeatureSamplingWithoutReplacementMixin,
                       options={OPTION_SAMPLE_SIZE, self.OPTION_NUM_RETAINED})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == NONE:
            config.use_no_feature_sampling()
        elif value == SAMPLING_WITHOUT_REPLACEMENT:
            c = config.use_feature_sampling_without_replacement()
            c.set_sample_size(options.get_float(OPTION_SAMPLE_SIZE, c.get_sample_size()))
            c.set_num_retained(options.get_int(self.OPTION_NUM_RETAINED, c.get_num_retained()))


class PartitionSamplingParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for creating a holdout set.
    """

    PARTITION_SAMPLING_RANDOM = 'random'

    OPTION_HOLDOUT_SET_SIZE = 'holdout_set_size'

    def __init__(self):
        super().__init__(name='holdout', description='The name of the strategy to be used for creating a holdout set')
        self.add_value(name=NONE, mixin=NoPartitionSamplingMixin)
        self.add_value(name=self.PARTITION_SAMPLING_RANDOM,
                       mixin=RandomBiPartitionSamplingMixin,
                       options={self.OPTION_HOLDOUT_SET_SIZE})
        self.add_value(name=SAMPLING_STRATIFIED_LABEL_WISE,
                       mixin=LabelWiseStratifiedBiPartitionSamplingMixin,
                       options={self.OPTION_HOLDOUT_SET_SIZE})
        self.add_value(name=SAMPLING_STRATIFIED_EXAMPLE_WISE,
                       mixin=ExampleWiseStratifiedBiPartitionSamplingMixin,
                       options={self.OPTION_HOLDOUT_SET_SIZE})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == NONE:
            config.use_no_partition_sampling()
        elif value == self.PARTITION_SAMPLING_RANDOM:
            c = config.use_random_bi_partition_sampling()
            c.set_holdout_set_size(options.get_float(self.OPTION_HOLDOUT_SET_SIZE, c.get_holdout_set_size()))
        elif value == SAMPLING_STRATIFIED_LABEL_WISE:
            c = config.use_label_wise_stratified_bi_partition_sampling()
            c.set_holdout_set_size(options.get_float(self.OPTION_HOLDOUT_SET_SIZE, c.get_holdout_set_size()))
        elif value == SAMPLING_STRATIFIED_EXAMPLE_WISE:
            c = config.use_example_wise_stratified_bi_partition_sampling()
            c.set_holdout_set_size(options.get_float(self.OPTION_HOLDOUT_SET_SIZE, c.get_holdout_set_size()))


class GlobalPruningParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for pruning entire rules.
    """

    GLOBAL_PRUNING_POST = 'post-pruning'

    OPTION_REMOVE_UNUSED_RULES = 'remove_unused_rules'

    OPTION_MIN_RULES = 'min_rules'

    OPTION_INTERVAL = 'interval'

    GLOBAL_PRUNING_PRE = 'pre-pruning'

    OPTION_AGGREGATION_FUNCTION = 'aggregation'

    OPTION_UPDATE_INTERVAL = 'update_interval'

    OPTION_STOP_INTERVAL = 'stop_interval'

    OPTION_NUM_PAST = 'num_past'

    OPTION_NUM_RECENT = 'num_recent'

    OPTION_MIN_IMPROVEMENT = 'min_improvement'

    AGGREGATION_FUNCTION_MIN = 'min'

    AGGREGATION_FUNCTION_MAX = 'max'

    AGGREGATION_FUNCTION_ARITHMETIC_MEAN = 'avg'

    def __init__(self):
        super().__init__(name='global_pruning',
                         description='The name of the strategy to be used for pruning entire rules')
        self.add_value(name=NONE, mixin=NoGlobalPruningMixin)
        self.add_value(name=self.GLOBAL_PRUNING_POST,
                       mixin=PostPruningMixin,
                       options={
                           OPTION_USE_HOLDOUT_SET, self.OPTION_REMOVE_UNUSED_RULES, self.OPTION_MIN_RULES,
                           self.OPTION_INTERVAL
                       })
        self.add_value(name=self.GLOBAL_PRUNING_PRE,
                       mixin=PrePruningMixin,
                       options={
                           OPTION_USE_HOLDOUT_SET, self.OPTION_REMOVE_UNUSED_RULES, self.OPTION_MIN_RULES,
                           self.OPTION_AGGREGATION_FUNCTION, self.OPTION_UPDATE_INTERVAL, self.OPTION_STOP_INTERVAL,
                           self.OPTION_NUM_PAST, self.OPTION_NUM_RECENT, self.OPTION_MIN_IMPROVEMENT
                       })

    def __create_aggregation_function(self, aggregation_function: str) -> AggregationFunction:
        value = parse_param(
            self.OPTION_AGGREGATION_FUNCTION, aggregation_function,
            {self.AGGREGATION_FUNCTION_MIN, self.AGGREGATION_FUNCTION_MAX, self.AGGREGATION_FUNCTION_ARITHMETIC_MEAN})

        if value == self.AGGREGATION_FUNCTION_MIN:
            return AggregationFunction.MIN
        elif value == self.AGGREGATION_FUNCTION_MAX:
            return AggregationFunction.MAX
        elif value == self.AGGREGATION_FUNCTION_ARITHMETIC_MEAN:
            return AggregationFunction.ARITHMETIC_MEAN

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == NONE:
            config.use_no_global_pruning()
        elif value == self.GLOBAL_PRUNING_POST:
            c = config.use_global_post_pruning()
            c.set_use_holdout_set(options.get_bool(OPTION_USE_HOLDOUT_SET, c.is_holdout_set_used()))
            c.set_remove_unused_rules(options.get_bool(self.OPTION_REMOVE_UNUSED_RULES, c.is_remove_unused_rules()))
            c.set_min_rules(options.get_int(self.OPTION_MIN_RULES, c.get_min_rules()))
            c.set_interval(options.get_int(self.OPTION_INTERVAL, c.get_interval()))
        elif value == self.GLOBAL_PRUNING_PRE:
            c = config.use_global_pre_pruning()
            c.set_use_holdout_set(options.get_bool(OPTION_USE_HOLDOUT_SET, c.is_holdout_set_used()))
            c.set_remove_unused_rules(options.get_bool(self.OPTION_REMOVE_UNUSED_RULES, c.is_remove_unused_rules()))
            c.set_min_rules(options.get_int(self.OPTION_MIN_RULES, c.get_min_rules()))
            aggregation_function = options.get_string(self.OPTION_AGGREGATION_FUNCTION, None)
            c.set_aggregation_function(
                self.__create_aggregation_function(aggregation_function) if aggregation_function is not None else c
                .get_aggregation_function())
            c.set_update_interval(options.get_int(self.OPTION_UPDATE_INTERVAL, c.get_update_interval()))
            c.set_stop_interval(options.get_int(self.OPTION_STOP_INTERVAL, c.get_stop_interval()))
            c.set_num_past(options.get_int(self.OPTION_NUM_PAST, c.get_num_past()))
            c.set_num_current(options.get_int(self.OPTION_NUM_RECENT, c.get_num_current()))
            c.set_min_improvement(options.get_float(self.OPTION_MIN_IMPROVEMENT, c.get_min_improvement()))


class RulePruningParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for pruning individual rules.
    """

    RULE_PRUNING_IREP = 'irep'

    def __init__(self):
        super().__init__(name='rule_pruning',
                         description='The name of the strategy to be used for pruning individual rules')
        self.add_value(name=NONE, mixin=NoRulePruningMixin)
        self.add_value(name=self.RULE_PRUNING_IREP, mixin=IrepRulePruningMixin)

    def _configure(self, config, value: str, _: Optional[Options]):
        if value == NONE:
            config.use_no_rule_pruning()
        elif value == self.RULE_PRUNING_IREP:
            config.use_irep_rule_pruning()


class ParallelRuleRefinementParameter(NominalParameter):
    """
    A parameter that allows to configure whether potential refinements of rules should be searched for in parallel or
    not.
    """

    def __init__(self):
        super().__init__(name='parallel_rule_refinement',
                         description='Whether potential refinements of rules should be searched for in parallel or not')
        self.add_value(name=BooleanOption.FALSE.value, mixin=NoParallelRuleRefinementMixin)
        self.add_value(name=BooleanOption.TRUE.value, mixin=ParallelRuleRefinementMixin, options={OPTION_NUM_THREADS})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == BooleanOption.FALSE.value:
            config.use_no_parallel_rule_refinement()
        else:
            c = config.use_parallel_rule_refinement()
            c.set_num_threads(options.get_int(OPTION_NUM_THREADS, c.get_num_threads()))


class ParallelStatisticUpdateParameter(NominalParameter):
    """
    A parameter that allows to configure whether the statistics for different examples should be updated in parallel or
    not.
    """

    def __init__(self):
        super().__init__(
            name='parallel_statistic_update',
            description='Whether the statistics for different examples should be updated in parallel or not')
        self.add_value(name=BooleanOption.FALSE.value, mixin=NoParallelStatisticUpdateMixin)
        self.add_value(name=BooleanOption.TRUE.value, mixin=ParallelStatisticUpdateMixin, options={OPTION_NUM_THREADS})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == BooleanOption.FALSE.value:
            config.use_no_parallel_statistic_update()
        else:
            c = config.use_parallel_statistic_update()
            c.set_num_threads(options.get_int(OPTION_NUM_THREADS, c.get_num_threads()))


class ParallelPredictionParameter(NominalParameter):
    """
    A parameter that allows to configure whether predictions for different examples should be obtained in parallel or
    not.
    """

    def __init__(self):
        super().__init__(name='parallel_prediction',
                         description='Whether predictions for different examples should be obtained in parallel or not')
        self.add_value(name=BooleanOption.FALSE.value, mixin=NoParallelPredictionMixin)
        self.add_value(name=BooleanOption.TRUE.value, mixin=ParallelPredictionMixin, options={OPTION_NUM_THREADS})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == BooleanOption.FALSE.value:
            config.use_no_parallel_prediction()
        else:
            c = config.use_parallel_prediction()
            c.set_num_threads(options.get_int(OPTION_NUM_THREADS, c.get_num_threads()))


class SizeStoppingCriterionParameter(IntParameter):
    """
    A parameter that allows to configure the maximum number of rules to be induced.
    """

    def __init__(self):
        super().__init__(
            name='max_rules',
            description='The maximum number of rules to be induced. Must be at least 1 or 0, if the number of rules '
            + 'should not be restricted',
            mixin=SizeStoppingCriterionMixin)

    def _configure(self, config, value):
        if value == 0 and issubclass(type(config), NoSizeStoppingCriterionMixin):
            config.use_no_size_stopping_criterion()
        else:
            config.use_size_stopping_criterion().set_max_rules(value)


class TimeStoppingCriterionParameter(IntParameter):
    """
    A parameter that allows to configure the duration in seconds after which the induction of rules should be canceled.
    """

    def __init__(self):
        super().__init__(
            name='time_limit',
            description='The duration in seconds after which the induction of rules should be canceled. Must be at '
            + 'least 1 or 0, if no time limit should be set',
            mixin=TimeStoppingCriterionMixin)

    def _configure(self, config, value):
        if value == 0 and issubclass(type(config), NoTimeStoppingCriterionMixin):
            config.use_no_time_stopping_criterion()
        else:
            config.use_time_stopping_criterion().set_time_limit(value)


class SequentialPostOptimizationParameter(NominalParameter):
    """
    A parameter that allows to configure whether each rule in a previously learned model should be optimized by being
    relearned in the context of the other rules or not.
    """

    OPTION_NUM_ITERATIONS = 'num_iterations'

    OPTION_REFINE_HEADS = 'refine_heads'

    def __init__(self):
        super().__init__(
            name='sequential_post_optimization',
            description='Whether each rule in a previously learned model should be optimized by being relearned in the '
            + 'context of the other rules or not')
        self.add_value(name=BooleanOption.FALSE.value, mixin=NoSequentialPostOptimizationMixin)
        self.add_value(name=BooleanOption.TRUE.value,
                       mixin=SequentialPostOptimizationMixin,
                       options={self.OPTION_NUM_ITERATIONS, self.OPTION_REFINE_HEADS, OPTION_RESAMPLE_FEATURES})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == BooleanOption.FALSE.value:
            config.use_no_sequential_post_optimization()
        elif value == BooleanOption.TRUE.value:
            c = config.use_sequential_post_optimization()
            c.set_num_iterations(options.get_int(self.OPTION_NUM_ITERATIONS, c.get_num_iterations()))
            c.set_refine_heads(options.get_bool(self.OPTION_REFINE_HEADS, c.are_heads_refined()))
            c.set_resample_features(options.get_bool(OPTION_RESAMPLE_FEATURES, c.are_features_resampled()))


RULE_LEARNER_PARAMETERS = {
    RuleInductionParameter(),
    FeatureBinningParameter(),
    LabelSamplingParameter(),
    InstanceSamplingParameter(),
    FeatureSamplingParameter(),
    PartitionSamplingParameter(),
    GlobalPruningParameter(),
    RulePruningParameter(),
    ParallelRuleRefinementParameter(),
    ParallelStatisticUpdateParameter(),
    ParallelPredictionParameter(),
    SizeStoppingCriterionParameter(),
    TimeStoppingCriterionParameter(),
    SequentialPostOptimizationParameter()
}


def configure_rule_learner(learner, config, parameters: Set[Parameter]):
    """
    Configures a rule learner by taking into account a given set of parameters.

    :param learner:     The rule learner to be configured
    :param config:      The configuration to be modified
    :param parameters:  A set that contains the parameters to be taken into account
    """
    for parameter in parameters:
        parameter_name = parameter.name

        if hasattr(learner, parameter_name):
            value = getattr(learner, parameter_name)

            if value is not None:
                parameter.configure(config=config, value=value)


def configure_argument_parser(parser: ArgumentParser, config_type: type, parameters: Set[Parameter]):
    """
    Configure an `ArgumentParser` by taking into account a given set of parameters.

    :param parser:      The `ArgumentParser` to be configured
    :param config_type: The type of the configuration that should support the parameters
    :param parameters:  A set that contains the parameters to be taken into account
    """
    for parameter in parameters:
        parameter.add_to_argument_parser(parser, config_type)


def create_kwargs_from_parameters(args, parameters: Set[Parameter]):
    """
    Creates and returns a dictionary that contains all parameter names and corresponding values that have been specified
    via an `ArgumentParser`.

    :param args:    The arguments that have been specified via an `ArgumentParser`
    :return:        A dictionary that contains the parameter names and corresponding values
    """
    kwargs = {}
    args_dict = vars(args)

    for parameter in parameters:
        parameter_name = parameter.name

        if parameter_name in args_dict:
            kwargs[parameter_name] = args_dict[parameter_name]

    return kwargs
