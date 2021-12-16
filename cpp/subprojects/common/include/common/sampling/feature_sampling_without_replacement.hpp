/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/feature_sampling.hpp"


/**
 * Allows to create instances of the type `IFeatureSampling` that select a random subset of the available features
 * without replacement.
 */
class FeatureSamplingWithoutReplacementFactory final : public IFeatureSamplingFactory {

    private:

        float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of features to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available features). Must be in (0, 1) or 0, if the default sample size
         *                   `floor(log2(num_features - 1) + 1)` should be used
         */
        FeatureSamplingWithoutReplacementFactory(float32 sampleSize);

        std::unique_ptr<IFeatureSampling> create(uint32 numFeatures) const override;

};
