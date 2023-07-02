/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/macros.hpp"
#include "common/sampling/feature_sampling.hpp"

/**
 * Defines an interface for all classes that allow to configure a method for sampling features without replacement.
 */
class MLRLCOMMON_API IFeatureSamplingWithoutReplacementConfig {
    public:

        virtual ~IFeatureSamplingWithoutReplacementConfig() {};

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

        /**
         * Returns the number of trailing features that are always included in a sample.
         *
         * @return The number of trailing features that are always included in a sample
         */
        virtual uint32 getNumRetained() const = 0;

        /**
         * Sets the number fo trailing features that should always be included in a sample.
         *
         * @param numRetained   The number of trailing features that should always be included in a sample. Must be at
         *                      least 0
         * @return              A reference to an object of type `IFeatureSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling features
         */
        virtual IFeatureSamplingWithoutReplacementConfig& setNumRetained(uint32 numRetained) = 0;
};

/**
 * Allows to configure a method for sampling features without replacement.
 */
class FeatureSamplingWithoutReplacementConfig final : public IFeatureSamplingConfig,
                                                      public IFeatureSamplingWithoutReplacementConfig {
    private:

        float32 sampleSize_;

        uint32 numRetained_;

    public:

        FeatureSamplingWithoutReplacementConfig();

        float32 getSampleSize() const override;

        IFeatureSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize) override;

        uint32 getNumRetained() const override;

        IFeatureSamplingWithoutReplacementConfig& setNumRetained(uint32 numRetained) override;

        std::unique_ptr<IFeatureSamplingFactory> createFeatureSamplingFactory(
          const IFeatureMatrix& featureMatrix) const override;

        bool isSamplingUsed() const override;
};
