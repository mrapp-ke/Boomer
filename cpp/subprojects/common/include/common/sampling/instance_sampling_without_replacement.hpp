/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/macros.hpp"
#include "common/sampling/instance_sampling.hpp"

/**
 * Defines an interface for all classes that allow to configure a method for selecting a subset of the available
 * training examples without replacement.
 */
class MLRLCOMMON_API IInstanceSamplingWithoutReplacementConfig {
    public:

        virtual ~IInstanceSamplingWithoutReplacementConfig() {};

        /**
         * Returns the fraction of examples that are included in a sample.
         *
         * @return The fraction of examples that are included in a sample
         */
        virtual float32 getSampleSize() const = 0;

        /**
         * Sets the fraction of examples that should be included in a sample.
         *
         * @param sampleSize    The fraction of examples that should be included in a sample, e.g., a value of 0.6
         *                      corresponds to 60 % of the available training examples. Must be in (0, 1)
         * @return              A reference to an object of type `IInstanceSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling instances
         */
        virtual IInstanceSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize) = 0;
};

/**
 * Allows to configure a method for selecting a subset of the available training examples without replacement.
 */
class InstanceSamplingWithoutReplacementConfig final : public IInstanceSamplingConfig,
                                                       public IInstanceSamplingWithoutReplacementConfig {
    private:

        float32 sampleSize_;

    public:

        InstanceSamplingWithoutReplacementConfig();

        float32 getSampleSize() const override;

        IInstanceSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize) override;

        std::unique_ptr<IInstanceSamplingFactory> createInstanceSamplingFactory() const override;
};
