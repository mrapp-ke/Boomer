/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/feature_sampling.hpp"


/**
 * Allows to create instances of the type `IFeatureSampling` that do not perform any sampling, but include all features.
 */
class NoFeatureSamplingFactory final : public IFeatureSamplingFactory {

    public:

        std::unique_ptr<IFeatureSampling> create(uint32 numFeatures) const override;

};
