/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/binning/feature_binning.hpp"
#include "common/multi_threading/multi_threading.hpp"

namespace boosting {

    /**
     * Allows to configure a method that automatically decides whether feature binning should be used or not.
     */
    class AutomaticFeatureBinningConfig final : public IFeatureBinningConfig {
        private:

            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

        public:

            /**
             * @param multiThreadingConfigPtr A reference to an unique pointer that stores the configuration of the
             *                                multi-threading behavior that should be used for the parallel update of
             *                                statistics
             */
            AutomaticFeatureBinningConfig(const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            /**
             * @see `IFeatureBinningConfig::createThresholdsFactory`
             */
            std::unique_ptr<IThresholdsFactory> createThresholdsFactory(const IFeatureMatrix& featureMatrix,
                                                                        const ILabelMatrix& labelMatrix) const override;
    };

}
