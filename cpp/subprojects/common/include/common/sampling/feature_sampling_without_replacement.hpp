/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/feature_sampling.hpp"
#include "common/macros.hpp"


/**
 * Defines an interface for all classes that allow to configure a method for sampling features without replacement.
 */
class MLRLCOMMON_API IFeatureSamplingWithoutReplacementConfig {

    public:

        virtual ~IFeatureSamplingWithoutReplacementConfig() { };

        /**
         * Returns the fraction of features that are included in a sample.
         *
         * @return The fraction of features that are included in a sample
         */
        virtual float32 getSampleSize() const = 0;

        /**
         * Sets the fraction of features that should be included in a sample.
         *
         * @param sampleSize    The fraction of features that should be included in a sample, e.g., a value of 0.6
         *                      corresponds to 60 % of the available features. Must be in (0, 1) or 0, if the default
         *                      sample size `floor(log2(numFeatures - 1) + 1)` should be used
         * @return              A reference to an object of type `IFeatureSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling features
         */
        virtual IFeatureSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize) = 0;

};

/**
 * Allows to configure a method for sampling features without replacement.
 */
class FeatureSamplingWithoutReplacementConfig final : public IFeatureSamplingConfig,
                                                      public IFeatureSamplingWithoutReplacementConfig {

    private:

        float32 sampleSize_;

    public:

        FeatureSamplingWithoutReplacementConfig();

        float32 getSampleSize() const override;

        IFeatureSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize) override;

        std::unique_ptr<IFeatureSamplingFactory> createFeatureSamplingFactory(
            const IFeatureMatrix& featureMatrix) const override;

};
