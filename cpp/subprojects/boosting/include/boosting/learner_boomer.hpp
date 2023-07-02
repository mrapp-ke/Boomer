/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "boosting/learner.hpp"

namespace boosting {

    /**
     * Defines the interface of the BOOMER algorithm.
     */
    class MLRLBOOSTING_API IBoomer : virtual public IBoostingRuleLearner {
        public:

            /**
             * Defines the interface for configuring the BOOMER algorithm.
             */
            class IConfig : virtual public IBoostingRuleLearner::IConfig,
                            virtual public IBoostingRuleLearner::IAutomaticPartitionSamplingMixin,
                            virtual public IBoostingRuleLearner::IAutomaticFeatureBinningMixin,
                            virtual public IBoostingRuleLearner::IAutomaticParallelRuleRefinementMixin,
                            virtual public IBoostingRuleLearner::IAutomaticParallelStatisticUpdateMixin,
                            virtual public IBoostingRuleLearner::IConstantShrinkageMixin,
                            virtual public IBoostingRuleLearner::INoL1RegularizationMixin,
                            virtual public IBoostingRuleLearner::IL1RegularizationMixin,
                            virtual public IBoostingRuleLearner::INoL2RegularizationMixin,
                            virtual public IBoostingRuleLearner::IL2RegularizationMixin,
                            virtual public IBoostingRuleLearner::INoDefaultRuleMixin,
                            virtual public IBoostingRuleLearner::IAutomaticDefaultRuleMixin,
                            virtual public IBoostingRuleLearner::ICompleteHeadMixin,
                            virtual public IBoostingRuleLearner::IDynamicPartialHeadMixin,
                            virtual public IBoostingRuleLearner::IFixedPartialHeadMixin,
                            virtual public IBoostingRuleLearner::ISingleLabelHeadMixin,
                            virtual public IBoostingRuleLearner::IAutomaticHeadMixin,
                            virtual public IBoostingRuleLearner::IDenseStatisticsMixin,
                            virtual public IBoostingRuleLearner::ISparseStatisticsMixin,
                            virtual public IBoostingRuleLearner::IAutomaticStatisticsMixin,
                            virtual public IBoostingRuleLearner::IExampleWiseLogisticLossMixin,
                            virtual public IBoostingRuleLearner::IExampleWiseSquaredErrorLossMixin,
                            virtual public IBoostingRuleLearner::IExampleWiseSquaredHingeLossMixin,
                            virtual public IBoostingRuleLearner::ILabelWiseLogisticLossMixin,
                            virtual public IBoostingRuleLearner::ILabelWiseSquaredErrorLossMixin,
                            virtual public IBoostingRuleLearner::ILabelWiseSquaredHingeLossMixin,
                            virtual public IBoostingRuleLearner::INoLabelBinningMixin,
                            virtual public IBoostingRuleLearner::IEqualWidthLabelBinningMixin,
                            virtual public IBoostingRuleLearner::IAutomaticLabelBinningMixin,
                            virtual public IBoostingRuleLearner::IIsotonicMarginalProbabilityCalibrationMixin,
                            virtual public IBoostingRuleLearner::IIsotonicJointProbabilityCalibrationMixin,
                            virtual public IBoostingRuleLearner::ILabelWiseBinaryPredictorMixin,
                            virtual public IBoostingRuleLearner::IExampleWiseBinaryPredictorMixin,
                            virtual public IBoostingRuleLearner::IGfmBinaryPredictorMixin,
                            virtual public IBoostingRuleLearner::IAutomaticBinaryPredictorMixin,
                            virtual public IBoostingRuleLearner::ILabelWiseScorePredictorMixin,
                            virtual public IBoostingRuleLearner::ILabelWiseProbabilityPredictorMixin,
                            virtual public IBoostingRuleLearner::IMarginalizedProbabilityPredictorMixin,
                            virtual public IBoostingRuleLearner::IAutomaticProbabilityPredictorMixin,
                            virtual public IRuleLearner::ISequentialRuleModelAssemblageMixin,
                            virtual public IRuleLearner::IDefaultRuleMixin,
                            virtual public IRuleLearner::IGreedyTopDownRuleInductionMixin,
                            virtual public IRuleLearner::IBeamSearchTopDownRuleInductionMixin,
                            virtual public IRuleLearner::INoPostProcessorMixin,
                            virtual public IRuleLearner::INoFeatureBinningMixin,
                            virtual public IRuleLearner::IEqualWidthFeatureBinningMixin,
                            virtual public IRuleLearner::IEqualFrequencyFeatureBinningMixin,
                            virtual public IRuleLearner::INoLabelSamplingMixin,
                            virtual public IRuleLearner::IRoundRobinLabelSamplingMixin,
                            virtual public IRuleLearner::ILabelSamplingWithoutReplacementMixin,
                            virtual public IRuleLearner::INoInstanceSamplingMixin,
                            virtual public IRuleLearner::IInstanceSamplingWithoutReplacementMixin,
                            virtual public IRuleLearner::IInstanceSamplingWithReplacementMixin,
                            virtual public IRuleLearner::ILabelWiseStratifiedInstanceSamplingMixin,
                            virtual public IRuleLearner::IExampleWiseStratifiedInstanceSamplingMixin,
                            virtual public IRuleLearner::INoFeatureSamplingMixin,
                            virtual public IRuleLearner::IFeatureSamplingWithoutReplacementMixin,
                            virtual public IRuleLearner::INoPartitionSamplingMixin,
                            virtual public IRuleLearner::IRandomBiPartitionSamplingMixin,
                            virtual public IRuleLearner::ILabelWiseStratifiedBiPartitionSamplingMixin,
                            virtual public IRuleLearner::IExampleWiseStratifiedBiPartitionSamplingMixin,
                            virtual public IRuleLearner::INoRulePruningMixin,
                            virtual public IRuleLearner::IIrepRulePruningMixin,
                            virtual public IRuleLearner::INoParallelRuleRefinementMixin,
                            virtual public IRuleLearner::IParallelRuleRefinementMixin,
                            virtual public IRuleLearner::INoParallelStatisticUpdateMixin,
                            virtual public IRuleLearner::IParallelStatisticUpdateMixin,
                            virtual public IRuleLearner::INoParallelPredictionMixin,
                            virtual public IRuleLearner::IParallelPredictionMixin,
                            virtual public IRuleLearner::INoSizeStoppingCriterionMixin,
                            virtual public IRuleLearner::ISizeStoppingCriterionMixin,
                            virtual public IRuleLearner::INoTimeStoppingCriterionMixin,
                            virtual public IRuleLearner::ITimeStoppingCriterionMixin,
                            virtual public IRuleLearner::IPrePruningMixin,
                            virtual public IRuleLearner::INoGlobalPruningMixin,
                            virtual public IRuleLearner::IPostPruningMixin,
                            virtual public IRuleLearner::INoSequentialPostOptimizationMixin,
                            virtual public IRuleLearner::ISequentialPostOptimizationMixin,
                            virtual public IRuleLearner::INoMarginalProbabilityCalibrationMixin,
                            virtual public IRuleLearner::INoJointProbabilityCalibrationMixin {
                public:

                    virtual ~IConfig() override {};
            };

            virtual ~IBoomer() override {};
    };

    /**
     * The BOOMER algorithm.
     */
    class Boomer final : public AbstractBoostingRuleLearner,
                         virtual public IBoomer {
        public:

            /**
             * Allows to configure the BOOMER algorithm.
             */
            class Config final : public AbstractBoostingRuleLearner::Config,
                                 virtual public IBoomer::IConfig {
                public:

                    Config();

                    /**
                     * @see `IRuleLearner::ISizeStoppingCriterionMixin::useSizeStoppingCriterion`
                     */
                    ISizeStoppingCriterionConfig& useSizeStoppingCriterion() override;
            };

        private:

            const std::unique_ptr<IBoomer::IConfig> configPtr_;

        public:

            /**
             * @param configPtr     An unique pointer to an object of type `IBoomer::IConfig` that specifies the
             *                      configuration that should be used by the rule learner
             * @param ddotFunction  A function pointer to BLAS' DDOT routine
             * @param dspmvFunction A function pointer to BLAS' DSPMV routine
             * @param dsysvFunction A function pointer to LAPACK'S DSYSV routine
             */
            Boomer(std::unique_ptr<IBoomer::IConfig> configPtr, Blas::DdotFunction ddotFunction,
                   Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction);
    };

    /**
     * Creates and returns a new object of type `IBoomer::IConfig`.
     *
     * @return An unique pointer to an object of type `IBoomer::IConfig` that has been created
     */
    MLRLBOOSTING_API std::unique_ptr<IBoomer::IConfig> createBoomerConfig();

    /**
     * Creates and returns a new object of type `IBoomer`.
     *
     * @param configPtr     An unique pointer to an object of type `IBoomer::IConfig` that specifies the configuration
     *                      that should be used by the rule learner
     * @param ddotFunction  A function pointer to BLAS' DDOT routine
     * @param dspmvFunction A function pointer to BLAS' DSPMV routine
     * @param dsysvFunction A function pointer to LAPACK'S DSYSV routine
     * @return              An unique pointer to an object of type `IBoomer` that has been created
     */
    MLRLBOOSTING_API std::unique_ptr<IBoomer> createBoomer(std::unique_ptr<IBoomer::IConfig> configPtr,
                                                           Blas::DdotFunction ddotFunction,
                                                           Blas::DspmvFunction dspmvFunction,
                                                           Lapack::DsysvFunction dsysvFunction);

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
