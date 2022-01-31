/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/feature_sampling.hpp"


/**
 * Allows to configure a method for sampling features that does not perform any sampling, but includes all features.
 */
class NoFeatureSamplingConfig final : public IFeatureSamplingConfig {

    public:

        std::unique_ptr<IFeatureSamplingFactory> createFeatureSamplingFactory(
            const IFeatureMatrix& featureMatrix) const override;

};
