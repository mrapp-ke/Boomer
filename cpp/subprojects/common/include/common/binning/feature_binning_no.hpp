/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/binning/feature_binning.hpp"
#include "common/multi_threading/multi_threading.hpp"

/**
 * Allows to configure a method that does not actually perform any feature binning.
 */
class NoFeatureBinningConfig final : public IFeatureBinningConfig {
    private:

        const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

    public:

        /**
         * @param multiThreadingConfigPtr A reference to an unique pointer that stores the configuration of the
         *                                multi-threading behavior that should be used for the parallel update of
         *                                statistics
         */
        NoFeatureBinningConfig(const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

        std::unique_ptr<IThresholdsFactory> createThresholdsFactory(const IFeatureMatrix& featureMatrix,
                                                                    const ILabelMatrix& labelMatrix) const override;
};
