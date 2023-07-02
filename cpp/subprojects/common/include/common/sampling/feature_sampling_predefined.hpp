/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/feature_sampling.hpp"

/**
 * An implementation of the class `IFeatureSampling` that does not perform any sampling, but always returns a predefined
 * set of features.
 */
class PredefinedFeatureSampling final : public IFeatureSampling {
    private:

        const IIndexVector& indexVector_;

    public:

        /**
         * @param indexVector A reference to an object of type `IIndexVector` that stores predefined feature indices
         */
        PredefinedFeatureSampling(const IIndexVector& indexVector);

        const IIndexVector& sample(RNG& rng) override;

        std::unique_ptr<IFeatureSampling> createBeamSearchFeatureSampling(RNG& rng, bool resample) override;
};
