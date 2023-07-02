
from libcpp.memory cimport unique_ptr

from mlrl.common.cython.learner cimport IBeamSearchTopDownRuleInductionMixin, IDefaultRuleMixin, \
    IEqualFrequencyFeatureBinningMixin, IEqualWidthFeatureBinningMixin, \
    IExampleWiseStratifiedBiPartitionSamplingMixin, IExampleWiseStratifiedInstanceSamplingMixin, \
    IFeatureSamplingWithoutReplacementMixin, IGreedyTopDownRuleInductionMixin, \
    IInstanceSamplingWithoutReplacementMixin, IInstanceSamplingWithReplacementMixin, IIrepRulePruningMixin, \
    ILabelSamplingWithoutReplacementMixin, ILabelWiseStratifiedBiPartitionSamplingMixin, \
    ILabelWiseStratifiedInstanceSamplingMixin, INoFeatureBinningMixin, INoFeatureSamplingMixin, INoGlobalPruningMixin, \
    INoInstanceSamplingMixin, INoJointProbabilityCalibrationMixin, INoLabelSamplingMixin, \
    INoMarginalProbabilityCalibrationMixin, INoParallelPredictionMixin, INoParallelRuleRefinementMixin, \
    INoParallelStatisticUpdateMixin, INoPartitionSamplingMixin, INoPostProcessorMixin, INoRulePruningMixin, \
    INoSequentialPostOptimizationMixin, INoSizeStoppingCriterionMixin, INoTimeStoppingCriterionMixin, \
    IParallelPredictionMixin, IParallelRuleRefinementMixin, IParallelStatisticUpdateMixin, IPostPruningMixin, \
    IPrePruningMixin, IRandomBiPartitionSamplingMixin, IRoundRobinLabelSamplingMixin, IRuleLearner, \
    ISequentialPostOptimizationMixin, ISequentialRuleModelAssemblageMixin, ISizeStoppingCriterionMixin, \
    ITimeStoppingCriterionMixin, RuleLearner, RuleLearnerConfig

from mlrl.boosting.cython.learner cimport DdotFunction, DspmvFunction, DsysvFunction, IAutomaticBinaryPredictorMixin, \
    IAutomaticDefaultRuleMixin, IAutomaticFeatureBinningMixin, IAutomaticHeadMixin, IAutomaticLabelBinningMixin, \
    IAutomaticParallelRuleRefinementMixin, IAutomaticParallelStatisticUpdateMixin, IAutomaticPartitionSamplingMixin, \
    IAutomaticProbabilityPredictorMixin, IAutomaticStatisticsMixin, ICompleteHeadMixin, IConstantShrinkageMixin, \
    IDenseStatisticsMixin, IDynamicPartialHeadMixin, IEqualWidthLabelBinningMixin, IExampleWiseBinaryPredictorMixin, \
    IExampleWiseLogisticLossMixin, IExampleWiseSquaredErrorLossMixin, IExampleWiseSquaredHingeLossMixin, \
    IFixedPartialHeadMixin, IGfmBinaryPredictorMixin, IIsotonicJointProbabilityCalibrationMixin, \
    IIsotonicMarginalProbabilityCalibrationMixin, IL1RegularizationMixin, IL2RegularizationMixin, \
    ILabelWiseBinaryPredictorMixin, ILabelWiseLogisticLossMixin, ILabelWiseProbabilityPredictorMixin, \
    ILabelWiseScorePredictorMixin, ILabelWiseSquaredErrorLossMixin, ILabelWiseSquaredHingeLossMixin, \
    IMarginalizedProbabilityPredictorMixin, INoDefaultRuleMixin, INoL1RegularizationMixin, INoL2RegularizationMixin, \
    INoLabelBinningMixin, ISingleLabelHeadMixin, ISparseStatisticsMixin


cdef extern from "boosting/learner_boomer.hpp" namespace "boosting" nogil:

    cdef cppclass IBoomerConfig"boosting::IBoomer::IConfig"(IAutomaticPartitionSamplingMixin,
                                                            IAutomaticFeatureBinningMixin,
                                                            IAutomaticParallelRuleRefinementMixin,
                                                            IAutomaticParallelStatisticUpdateMixin,
                                                            IConstantShrinkageMixin,
                                                            INoL1RegularizationMixin,
                                                            IL1RegularizationMixin,
                                                            INoL2RegularizationMixin,
                                                            IL2RegularizationMixin,
                                                            INoDefaultRuleMixin,
                                                            IAutomaticDefaultRuleMixin,
                                                            ICompleteHeadMixin,
                                                            IDynamicPartialHeadMixin,
                                                            IFixedPartialHeadMixin,
                                                            ISingleLabelHeadMixin,
                                                            IAutomaticHeadMixin,
                                                            IDenseStatisticsMixin,
                                                            ISparseStatisticsMixin,
                                                            IAutomaticStatisticsMixin,
                                                            IExampleWiseLogisticLossMixin,
                                                            IExampleWiseSquaredErrorLossMixin,
                                                            IExampleWiseSquaredHingeLossMixin,
                                                            ILabelWiseLogisticLossMixin,
                                                            ILabelWiseSquaredErrorLossMixin,
                                                            ILabelWiseSquaredHingeLossMixin,
                                                            INoLabelBinningMixin,
                                                            IEqualWidthLabelBinningMixin,
                                                            IAutomaticLabelBinningMixin,
                                                            IIsotonicMarginalProbabilityCalibrationMixin,
                                                            IIsotonicJointProbabilityCalibrationMixin,
                                                            ILabelWiseBinaryPredictorMixin,
                                                            IExampleWiseBinaryPredictorMixin,
                                                            IGfmBinaryPredictorMixin,
                                                            IAutomaticBinaryPredictorMixin,
                                                            ILabelWiseScorePredictorMixin,
                                                            ILabelWiseProbabilityPredictorMixin,
                                                            IMarginalizedProbabilityPredictorMixin,
                                                            IAutomaticProbabilityPredictorMixin,
                                                            ISequentialRuleModelAssemblageMixin,
                                                            IDefaultRuleMixin,
                                                            IGreedyTopDownRuleInductionMixin,
                                                            IBeamSearchTopDownRuleInductionMixin,
                                                            INoPostProcessorMixin,
                                                            INoFeatureBinningMixin,
                                                            IEqualWidthFeatureBinningMixin,
                                                            IEqualFrequencyFeatureBinningMixin,
                                                            INoLabelSamplingMixin,
                                                            IRoundRobinLabelSamplingMixin,
                                                            ILabelSamplingWithoutReplacementMixin,
                                                            INoInstanceSamplingMixin,
                                                            IInstanceSamplingWithoutReplacementMixin,
                                                            IInstanceSamplingWithReplacementMixin,
                                                            ILabelWiseStratifiedInstanceSamplingMixin,
                                                            IExampleWiseStratifiedInstanceSamplingMixin,
                                                            INoFeatureSamplingMixin,
                                                            IFeatureSamplingWithoutReplacementMixin,
                                                            INoPartitionSamplingMixin,
                                                            IRandomBiPartitionSamplingMixin,
                                                            ILabelWiseStratifiedBiPartitionSamplingMixin,
                                                            IExampleWiseStratifiedBiPartitionSamplingMixin,
                                                            INoRulePruningMixin,
                                                            IIrepRulePruningMixin,
                                                            INoParallelRuleRefinementMixin,
                                                            IParallelRuleRefinementMixin,
                                                            INoParallelStatisticUpdateMixin,
                                                            IParallelStatisticUpdateMixin,
                                                            INoParallelPredictionMixin,
                                                            IParallelPredictionMixin,
                                                            INoSizeStoppingCriterionMixin,
                                                            ISizeStoppingCriterionMixin,
                                                            INoTimeStoppingCriterionMixin,
                                                            ITimeStoppingCriterionMixin,
                                                            IPrePruningMixin,
                                                            INoGlobalPruningMixin,
                                                            IPostPruningMixin,
                                                            INoSequentialPostOptimizationMixin,
                                                            ISequentialPostOptimizationMixin,
                                                            INoMarginalProbabilityCalibrationMixin,
                                                            INoJointProbabilityCalibrationMixin):
        pass

    cdef cppclass IBoomer(IRuleLearner):
        pass


    unique_ptr[IBoomerConfig] createBoomerConfig()


    unique_ptr[IBoomer] createBoomer(unique_ptr[IBoomerConfig] configPtr, DdotFunction ddotFunction,
                                     DspmvFunction dspmvFunction, DsysvFunction dsysvFunction)


cdef class BoomerConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IBoomerConfig] config_ptr


cdef class Boomer(RuleLearner):

    # Attributes:

    cdef unique_ptr[IBoomer] rule_learner_ptr
