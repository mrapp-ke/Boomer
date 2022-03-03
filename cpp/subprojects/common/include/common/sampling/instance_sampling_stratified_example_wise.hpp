/*
 * @author Anna Kulischkin (Anna_Kulischkin@web.de)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/instance_sampling.hpp"
#include "common/macros.hpp"


/**
 * Defines an interface for all classes that allow to configure a method for selecting a subset of the available
 * training examples using stratification, where distinct label vectors are treated as individual classes.
 */
class MLRLCOMMON_API IExampleWiseStratifiedInstanceSamplingConfig {

    public:

        virtual ~IExampleWiseStratifiedInstanceSamplingConfig() { };

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
         * @return              A reference to an object of type `IExampleWiseStratifiedInstanceSamplingConfig` that
         *                      allows further configuration of the method for sampling instances
         */
        virtual IExampleWiseStratifiedInstanceSamplingConfig& setSampleSize(float32 sampleSize) = 0;

};

/**
 * Allows to configure a method for selecting a subset of the available training examples using stratification, where
 * distinct label vectors are treated as individual classes.
 */
class ExampleWiseStratifiedInstanceSamplingConfig final : public IInstanceSamplingConfig,
                                                          public IExampleWiseStratifiedInstanceSamplingConfig {

    private:

        float32 sampleSize_;

    public:

        ExampleWiseStratifiedInstanceSamplingConfig();

        float32 getSampleSize() const override;

        IExampleWiseStratifiedInstanceSamplingConfig& setSampleSize(float32 sampleSize) override;

        std::unique_ptr<IInstanceSamplingFactory> createInstanceSamplingFactory() const override;

};
