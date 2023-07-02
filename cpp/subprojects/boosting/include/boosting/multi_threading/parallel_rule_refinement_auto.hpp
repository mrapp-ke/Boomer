/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss.hpp"
#include "boosting/rule_evaluation/head_type.hpp"
#include "common/multi_threading/multi_threading.hpp"
#include "common/sampling/feature_sampling.hpp"

namespace boosting {

    /**
     * Allows to configure the multi-threading behavior that is used for the parallel refinement of rules by
     * automatically deciding for the number of threads to be used.
     */
    class AutoParallelRuleRefinementConfig final : public IMultiThreadingConfig {
        private:

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

            const std::unique_ptr<IHeadConfig>& headConfigPtr_;

            const std::unique_ptr<IFeatureSamplingConfig>& featureSamplingConfigPtr_;

        public:

            /**
             * @param lossConfigPtr             A reference to an unique pointer that stores the configuration of the
             *                                  loss function
             * @param headConfigPtr             A reference to an unique pointer that stores the configuration of rule
             *                                  heads
             * @param featureSamplingConfigPtr  A reference to an unique pointer that stores the configuration of the
             *                                  method for sampling features
             */
            AutoParallelRuleRefinementConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr,
                                             const std::unique_ptr<IHeadConfig>& headConfigPtr,
                                             const std::unique_ptr<IFeatureSamplingConfig>& featureSamplingConfigPtr);

            /**
             * @see `IMultiThreadingConfig::getNumThreads`
             */
            uint32 getNumThreads(const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;
    };

}
