"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility function for configuring boosting algorithms.
"""
from typing import Optional

from mlrl.common.config import AUTOMATIC, BINNING_EQUAL_WIDTH, NONE, OPTION_BIN_RATIO, OPTION_MAX_BINS, \
    OPTION_MIN_BINS, OPTION_USE_HOLDOUT_SET, RULE_LEARNER_PARAMETERS, FeatureBinningParameter, FloatParameter, \
    NominalParameter, ParallelRuleRefinementParameter, ParallelStatisticUpdateParameter, PartitionSamplingParameter
from mlrl.common.cython.learner import DefaultRuleMixin, NoJointProbabilityCalibrationMixin, \
    NoMarginalProbabilityCalibrationMixin, NoPostProcessorMixin
from mlrl.common.options import BooleanOption, Options

from mlrl.boosting.cython.learner import AutomaticBinaryPredictorMixin, AutomaticDefaultRuleMixin, \
    AutomaticFeatureBinningMixin, AutomaticLabelBinningMixin, AutomaticParallelRuleRefinementMixin, \
    AutomaticParallelStatisticUpdateMixin, AutomaticPartitionSamplingMixin, AutomaticProbabilityPredictorMixin, \
    AutomaticStatisticsMixin, CompleteHeadMixin, ConstantShrinkageMixin, DenseStatisticsMixin, \
    DynamicPartialHeadMixin, EqualWidthLabelBinningMixin, ExampleWiseBinaryPredictorMixin, \
    ExampleWiseLogisticLossMixin, ExampleWiseSquaredErrorLossMixin, ExampleWiseSquaredHingeLossMixin, \
    FixedPartialHeadMixin, GfmBinaryPredictorMixin, IsotonicJointProbabilityCalibrationMixin, \
    IsotonicMarginalProbabilityCalibrationMixin, L1RegularizationMixin, L2RegularizationMixin, \
    LabelWiseBinaryPredictorMixin, LabelWiseLogisticLossMixin, LabelWiseProbabilityPredictorMixin, \
    LabelWiseSquaredErrorLossMixin, LabelWiseSquaredHingeLossMixin, MarginalizedProbabilityPredictorMixin, \
    NoDefaultRuleMixin, NoL1RegularizationMixin, NoL2RegularizationMixin, NoLabelBinningMixin, SingleLabelHeadMixin, \
    SparseStatisticsMixin

PROBABILITY_CALIBRATION_ISOTONIC = 'isotonic'

OPTION_BASED_ON_PROBABILITIES = 'based_on_probabilities'

OPTION_USE_PROBABILITY_CALIBRATION_MODEL = 'use_probability_calibration'


class ExtendedPartitionSamplingParameter(PartitionSamplingParameter):
    """
    Extends the `PartitionSamplingParameter` by a value for automatic configuration.
    """

    def __init__(self):
        super().__init__()
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticPartitionSamplingMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'depending on whether a holdout set is needed and depending on the loss function')

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == AUTOMATIC:
            config.use_automatic_partition_sampling()
        else:
            super()._configure(config, value, options)


class ExtendedFeatureBinningParameter(FeatureBinningParameter):
    """
    Extends the `FeatureBinningParameter` by a value for automatic configuration.
    """

    def __init__(self):
        super().__init__()
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticFeatureBinningMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'based on the characteristics of the feature matrix')

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == AUTOMATIC:
            config.use_automatic_feature_binning()
        else:
            super()._configure(config, value, options)


class ExtendedParallelRuleRefinementParameter(ParallelRuleRefinementParameter):
    """
    Extends the `ParallelRuleRefinementParameter` by a value for automatic configuration.
    """

    def __init__(self):
        super().__init__()
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticParallelRuleRefinementMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'based on ' + 'the parameter ' + LossParameter().argument_name)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == AUTOMATIC:
            config.use_automatic_parallel_rule_refinement()
        else:
            super()._configure(config, value, options)


class ExtendedParallelStatisticUpdateParameter(ParallelStatisticUpdateParameter):
    """
    Extends the `ParallelStatisticUpdateParameter` by a value for automatic configuration.
    """

    def __init__(self):
        super().__init__()
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticParallelStatisticUpdateMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'based on ' + 'the parameter ' + LossParameter().argument_name)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == AUTOMATIC:
            config.use_automatic_parallel_statistic_update()
        else:
            super()._configure(config, value, options)


class ShrinkageParameter(FloatParameter):
    """
    A parameter that allows to configure the shrinkage parameter, a.k.a. the learning rate, to be used.
    """

    def __init__(self):
        super().__init__(
            name='shrinkage',
            description='The shrinkage parameter, a.k.a. the learning rate, to be used. Must be in (0, 1].',
            mixin=ConstantShrinkageMixin)

    def _configure(self, config, value):
        if value == 1.0 and issubclass(type(config), NoPostProcessorMixin):
            config.use_no_post_processor()
        else:
            config.use_constant_shrinkage_post_processor().set_shrinkage(value)


class L1RegularizationParameter(FloatParameter):
    """
    A parameter that allows to configure the weight of the L1 regularization.
    """

    def __init__(self):
        super().__init__(name='l1_regularization_weight',
                         description='The weight of the L1 regularization. Must be at least 0',
                         mixin=L1RegularizationMixin)

    def _configure(self, config, value):
        if value == 0.0 and issubclass(type(config), NoL1RegularizationMixin):
            config.use_no_l1_regularization()
        else:
            config.use_l1_regularization().set_regularization_weight(value)


class L2RegularizationParameter(FloatParameter):
    """
    A parameter that allows to configure the weight of the L2 regularization.
    """

    def __init__(self):
        super().__init__(name='l2_regularization_weight',
                         description='The weight of the L2 regularization. Must be at least 0',
                         mixin=L2RegularizationMixin)

    def _configure(self, config, value):
        if value == 0.0 and issubclass(type(config), NoL2RegularizationMixin):
            config.use_no_l2_regularization()
        else:
            config.use_l2_regularization().set_regularization_weight(value)


class DefaultRuleParameter(NominalParameter):
    """
    A parameter that allows to configure whether a default rule should be induced or not.
    """

    def __init__(self):
        super().__init__(name='default_rule', description='Whether a default rule should be induced or not')
        self.add_value(name=BooleanOption.FALSE.value, mixin=NoDefaultRuleMixin)
        self.add_value(name=BooleanOption.TRUE.value, mixin=DefaultRuleMixin)
        self.add_value(name=AUTOMATIC, mixin=AutomaticDefaultRuleMixin)

    def _configure(self, config, value: str, _: Optional[Options]):
        if value == BooleanOption.FALSE.value:
            config.use_no_default_rule()
        elif value == BooleanOption.TRUE.value:
            config.use_default_rule()
        elif value == AUTOMATIC:
            config.use_automatic_default_rule()


class StatisticFormatParameter(NominalParameter):
    """
    A parameter that allows to configure the format to be used for the representation of gradients and Hessians.
    """

    STATISTIC_FORMAT_DENSE = 'dense'

    STATISTIC_FORMAT_SPARSE = 'sparse'

    def __init__(self):
        super().__init__(name='statistic_format',
                         description='The format to be used for the representation of gradients and Hessians')
        self.add_value(name=self.STATISTIC_FORMAT_DENSE, mixin=DenseStatisticsMixin)
        self.add_value(name=self.STATISTIC_FORMAT_SPARSE, mixin=SparseStatisticsMixin)
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticStatisticsMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable format is chosen automatically '
                       + 'based on the parameters ' + LossParameter().argument_name + ', '
                       + HeadTypeParameter().argument_name + ', ' + DefaultRuleParameter().argument_name + ' and the '
                       + 'characteristics of the label matrix')

    def _configure(self, config, value: str, _: Optional[Options]):
        if value == self.STATISTIC_FORMAT_DENSE:
            config.use_dense_statistics()
        elif value == self.STATISTIC_FORMAT_SPARSE:
            config.use_sparse_statistics()
        elif value == AUTOMATIC:
            config.use_automatic_statistics()


class LabelBinningParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for gradient-based label binning (GBLB).
    """

    def __init__(self):
        super().__init__(name='label_binning',
                         description='The name of the strategy to be used for gradient-based label binning (GBLB)')
        self.add_value(name=NONE, mixin=NoLabelBinningMixin)
        self.add_value(name=BINNING_EQUAL_WIDTH,
                       mixin=EqualWidthLabelBinningMixin,
                       options={OPTION_BIN_RATIO, OPTION_MIN_BINS, OPTION_MAX_BINS})
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticLabelBinningMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'based on the parameters ' + LossParameter().argument_name + ' and '
                       + HeadTypeParameter().argument_name)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == NONE:
            config.use_no_label_binning()
        elif value == BINNING_EQUAL_WIDTH:
            c = config.use_equal_width_label_binning()
            c.set_bin_ratio(options.get_float(OPTION_BIN_RATIO, c.get_bin_ratio()))
            c.set_min_bins(options.get_int(OPTION_MIN_BINS, c.get_min_bins()))
            c.set_max_bins(options.get_int(OPTION_MAX_BINS, c.get_max_bins()))
        elif value == AUTOMATIC:
            config.use_automatic_label_binning()


class LossParameter(NominalParameter):
    """
    A parameter that allows to configure the loss function to be minimized during training.
    """

    LOSS_LOGISTIC_LABEL_WISE = 'logistic-label-wise'

    LOSS_LOGISTIC_EXAMPLE_WISE = 'logistic-example-wise'

    LOSS_SQUARED_ERROR_LABEL_WISE = 'squared-error-label-wise'

    LOSS_SQUARED_ERROR_EXAMPLE_WISE = 'squared-error-example-wise'

    LOSS_SQUARED_HINGE_LABEL_WISE = 'squared-hinge-label-wise'

    LOSS_SQUARED_HINGE_EXAMPLE_WISE = 'squared-hinge-example-wise'

    def __init__(self):
        super().__init__(name='loss', description='The name of the loss function to be minimized during training')
        self.add_value(name=self.LOSS_LOGISTIC_LABEL_WISE, mixin=LabelWiseLogisticLossMixin)
        self.add_value(name=self.LOSS_LOGISTIC_EXAMPLE_WISE, mixin=ExampleWiseLogisticLossMixin)
        self.add_value(name=self.LOSS_SQUARED_ERROR_LABEL_WISE, mixin=LabelWiseSquaredErrorLossMixin)
        self.add_value(name=self.LOSS_SQUARED_ERROR_EXAMPLE_WISE, mixin=ExampleWiseSquaredErrorLossMixin)
        self.add_value(name=self.LOSS_SQUARED_HINGE_LABEL_WISE, mixin=LabelWiseSquaredHingeLossMixin)
        self.add_value(name=self.LOSS_SQUARED_HINGE_EXAMPLE_WISE, mixin=ExampleWiseSquaredHingeLossMixin)

    def _configure(self, config, value: str, _: Optional[Options]):
        if value == self.LOSS_LOGISTIC_LABEL_WISE:
            config.use_label_wise_logistic_loss()
        elif value == self.LOSS_LOGISTIC_EXAMPLE_WISE:
            config.use_example_wise_logistic_loss()
        elif value == self.LOSS_SQUARED_ERROR_LABEL_WISE:
            config.use_label_wise_squared_error_loss()
        elif value == self.LOSS_SQUARED_ERROR_EXAMPLE_WISE:
            config.use_example_wise_squared_error_loss()
        elif value == self.LOSS_SQUARED_HINGE_LABEL_WISE:
            config.use_label_wise_squared_hinge_loss()
        elif value == self.LOSS_SQUARED_HINGE_EXAMPLE_WISE:
            config.use_example_wise_squared_hinge_loss()


class HeadTypeParameter(NominalParameter):
    """
    A parameter that allows to configure the type of the rule heads that should be used.
    """

    HEAD_TYPE_SINGLE = 'single-label'

    HEAD_TYPE_PARTIAL_FIXED = 'partial-fixed'

    OPTION_LABEL_RATIO = 'label_ratio'

    OPTION_MIN_LABELS = 'min_labels'

    OPTION_MAX_LABELS = 'max_labels'

    HEAD_TYPE_PARTIAL_DYNAMIC = 'partial-dynamic'

    OPTION_THRESHOLD = 'threshold'

    OPTION_EXPONENT = 'exponent'

    HEAD_TYPE_COMPLETE = 'complete'

    def __init__(self):
        super().__init__(name='head_type', description='The type of the rule heads that should be used')
        self.add_value(name=self.HEAD_TYPE_SINGLE, mixin=SingleLabelHeadMixin)
        self.add_value(name=self.HEAD_TYPE_PARTIAL_FIXED,
                       mixin=FixedPartialHeadMixin,
                       options={self.OPTION_LABEL_RATIO, self.OPTION_MIN_LABELS, self.OPTION_MAX_LABELS})
        self.add_value(name=self.HEAD_TYPE_PARTIAL_DYNAMIC,
                       mixin=DynamicPartialHeadMixin,
                       options={self.OPTION_THRESHOLD, self.OPTION_EXPONENT})
        self.add_value(name=self.HEAD_TYPE_COMPLETE, mixin=CompleteHeadMixin)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == self.HEAD_TYPE_SINGLE:
            config.use_single_label_heads()
        elif value == self.HEAD_TYPE_PARTIAL_FIXED:
            c = config.use_fixed_partial_heads()
            c.set_label_ratio(options.get_float(self.OPTION_LABEL_RATIO, c.get_label_ratio()))
            c.set_min_labels(options.get_int(self.OPTION_MIN_LABELS, c.get_min_labels()))
            c.set_max_labels(options.get_int(self.OPTION_MAX_LABELS, c.get_max_labels()))
        elif value == self.HEAD_TYPE_PARTIAL_DYNAMIC:
            c = config.use_dynamic_partial_heads()
            c.set_threshold(options.get_float(self.OPTION_THRESHOLD, c.get_threshold()))
            c.set_exponent(options.get_float(self.OPTION_EXPONENT, c.get_exponent()))
        elif value == self.HEAD_TYPE_COMPLETE:
            config.use_complete_heads()
        elif value == AUTOMATIC:
            config.use_automatic_heads()


class MarginalProbabilityCalibrationParameter(NominalParameter):
    """
    A parameter that allows to configure the method to be used for the calibration of marginal probabilities.
    """

    def __init__(self):
        super().__init__(name='marginal_probability_calibration',
                         description='The name of the method to be used for the calibration of marginal probabilities')
        self.add_value(name=NONE, mixin=NoMarginalProbabilityCalibrationMixin)
        self.add_value(name=PROBABILITY_CALIBRATION_ISOTONIC,
                       mixin=IsotonicMarginalProbabilityCalibrationMixin,
                       options={OPTION_USE_HOLDOUT_SET})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == NONE:
            config.use_no_marginal_probability_calibration()
        if value == PROBABILITY_CALIBRATION_ISOTONIC:
            c = config.use_isotonic_marginal_probability_calibration()
            c.set_use_holdout_set(options.get_bool(OPTION_USE_HOLDOUT_SET, c.is_holdout_set_used()))


class JointProbabilityCalibrationParameter(NominalParameter):
    """
    A parameter that allows to configure the method to be used for the calibration of joint probabilities.
    """

    def __init__(self):
        super().__init__(name='joint_probability_calibration',
                         description='The name of the method to be used for the calibration of joint probabilities')
        self.add_value(name=NONE, mixin=NoJointProbabilityCalibrationMixin)
        self.add_value(name=PROBABILITY_CALIBRATION_ISOTONIC,
                       mixin=IsotonicJointProbabilityCalibrationMixin,
                       options={OPTION_USE_HOLDOUT_SET})

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == NONE:
            config.use_no_joint_probability_calibration()
        if value == PROBABILITY_CALIBRATION_ISOTONIC:
            c = config.use_isotonic_joint_probability_calibration()
            c.set_use_holdout_set(options.get_bool(OPTION_USE_HOLDOUT_SET, c.is_holdout_set_used()))


class BinaryPredictorParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for predicting binary labels.
    """

    BINARY_PREDICTOR_LABEL_WISE = 'label-wise'

    BINARY_PREDICTOR_EXAMPLE_WISE = 'example-wise'

    BINARY_PREDICTOR_GFM = 'gfm'

    def __init__(self):
        super().__init__(name='binary_predictor',
                         description='The name of the strategy to be used for predicting binary labels')
        self.add_value(name=self.BINARY_PREDICTOR_LABEL_WISE,
                       mixin=LabelWiseBinaryPredictorMixin,
                       options={OPTION_BASED_ON_PROBABILITIES, OPTION_USE_PROBABILITY_CALIBRATION_MODEL})
        self.add_value(name=self.BINARY_PREDICTOR_EXAMPLE_WISE,
                       mixin=ExampleWiseBinaryPredictorMixin,
                       options={OPTION_BASED_ON_PROBABILITIES, OPTION_USE_PROBABILITY_CALIBRATION_MODEL})
        self.add_value(name=self.BINARY_PREDICTOR_GFM,
                       mixin=GfmBinaryPredictorMixin,
                       options={OPTION_USE_PROBABILITY_CALIBRATION_MODEL})
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticBinaryPredictorMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'based on the parameter ' + LossParameter().argument_name)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == self.BINARY_PREDICTOR_LABEL_WISE:
            c = config.use_label_wise_binary_predictor()
            c.set_based_on_probabilities(options.get_bool(OPTION_BASED_ON_PROBABILITIES, c.is_based_on_probabilities()))
            c.set_use_probability_calibration_model(
                options.get_bool(OPTION_USE_PROBABILITY_CALIBRATION_MODEL, c.is_probability_calibration_model_used()))
        elif value == self.BINARY_PREDICTOR_EXAMPLE_WISE:
            c = config.use_example_wise_binary_predictor()
            c.set_based_on_probabilities(options.get_bool(OPTION_BASED_ON_PROBABILITIES, c.is_based_on_probabilities()))
            c.set_use_probability_calibration_model(
                options.get_bool(OPTION_USE_PROBABILITY_CALIBRATION_MODEL, c.is_probability_calibration_model_used()))
        elif value == self.BINARY_PREDICTOR_GFM:
            c = config.use_gfm_binary_predictor()
            c.set_use_probability_calibration_model(
                options.get_bool(OPTION_USE_PROBABILITY_CALIBRATION_MODEL, c.is_probability_calibration_model_used()))
        elif value == AUTOMATIC:
            config.use_automatic_binary_predictor()


class ProbabilityPredictorParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for predicting probabilities.
    """

    PROBABILITY_PREDICTOR_LABEL_WISE = 'label-wise'

    PROBABILITY_PREDICTOR_MARGINALIZED = 'marginalized'

    def __init__(self):
        super().__init__(name='probability_predictor',
                         description='The name of the strategy to be used for predicting probabilities')
        self.add_value(name=self.PROBABILITY_PREDICTOR_LABEL_WISE,
                       mixin=LabelWiseProbabilityPredictorMixin,
                       options={OPTION_USE_PROBABILITY_CALIBRATION_MODEL})
        self.add_value(name=self.PROBABILITY_PREDICTOR_MARGINALIZED,
                       mixin=MarginalizedProbabilityPredictorMixin,
                       options={OPTION_USE_PROBABILITY_CALIBRATION_MODEL})
        self.add_value(name=AUTOMATIC,
                       mixin=AutomaticProbabilityPredictorMixin,
                       description='If set to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically '
                       + 'based on the parameter ' + LossParameter().argument_name)

    def _configure(self, config, value: str, options: Optional[Options]):
        if value == self.PROBABILITY_PREDICTOR_LABEL_WISE:
            c = config.use_label_wise_probability_predictor()
            c.set_use_probability_calibration_model(
                options.get_bool(OPTION_USE_PROBABILITY_CALIBRATION_MODEL, c.is_probability_calibration_model_used()))
        elif value == self.PROBABILITY_PREDICTOR_MARGINALIZED:
            c = config.use_marginalized_probability_predictor()
            c.set_use_probability_calibration_model(
                options.get_bool(OPTION_USE_PROBABILITY_CALIBRATION_MODEL, c.is_probability_calibration_model_used()))
        elif value == AUTOMATIC:
            config.use_automatic_probability_predictor()


BOOSTING_RULE_LEARNER_PARAMETERS = RULE_LEARNER_PARAMETERS | {
    ExtendedPartitionSamplingParameter(),
    ExtendedFeatureBinningParameter(),
    ExtendedParallelRuleRefinementParameter(),
    ExtendedParallelStatisticUpdateParameter(),
    ShrinkageParameter(),
    L1RegularizationParameter(),
    L2RegularizationParameter(),
    DefaultRuleParameter(),
    StatisticFormatParameter(),
    LabelBinningParameter(),
    LossParameter(),
    HeadTypeParameter(),
    MarginalProbabilityCalibrationParameter(),
    JointProbabilityCalibrationParameter(),
    BinaryPredictorParameter(),
    ProbabilityPredictorParameter()
}
