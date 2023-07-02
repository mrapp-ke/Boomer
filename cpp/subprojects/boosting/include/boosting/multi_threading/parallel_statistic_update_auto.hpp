/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss.hpp"
#include "common/multi_threading/multi_threading.hpp"

namespace boosting {

    /**
     * Allows to configure the multi-threading behavior that is used for the parallel update of statistics by
     * automatically deciding for the number of threads to be used.
     */
    class AutoParallelStatisticUpdateConfig final : public IMultiThreadingConfig {
        private:

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

        public:

            /**
             * @param lossConfigPtr A reference to an unique pointer that stores the configuration of the loss function
             */
            AutoParallelStatisticUpdateConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr);

            /**
             * @see `IMultiThreadingConfig::getNumThreads`
             */
            uint32 getNumThreads(const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;
    };

}
