/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/feature_sampling.hpp"


/**
 * An implementation of the class `IFeatureSubSampling` that does not perform any sampling, but includes all features.
 */
class NoFeatureSubSampling final : public IFeatureSubSampling {

    public:

        std::unique_ptr<IIndexVector> subSample(uint32 numFeatures, RNG& rng) const override;

};
