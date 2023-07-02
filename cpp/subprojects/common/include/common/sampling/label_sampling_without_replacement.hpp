/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/macros.hpp"
#include "common/sampling/label_sampling.hpp"

/**
 * Defines an interface for all classes that allow to configure a method for sampling labels without replacement.
 */
class MLRLCOMMON_API ILabelSamplingWithoutReplacementConfig {
    public:

        virtual ~ILabelSamplingWithoutReplacementConfig() {};

        /**
         * Returns the number of labels that are included in a sample.
         *
         * @return The number of labels that are included in a sample
         */
        virtual uint32 getNumSamples() const = 0;

        /**
         * Sets the number of labels that should be included in a sample.
         *
         * @param numSamples    The number of labels that should be included in a sample. Must be at least 1
         * @return              A reference to an object of type `ILabelSamplingWithoutReplacementConfig` that allows
         *                      further configuration of the method for sampling labels
         */
        virtual ILabelSamplingWithoutReplacementConfig& setNumSamples(uint32 numSamples) = 0;
};

/**
 * Allows to configure a method for sampling labels without replacement.
 */
class LabelSamplingWithoutReplacementConfig final : public ILabelSamplingConfig,
                                                    public ILabelSamplingWithoutReplacementConfig {
    private:

        uint32 numSamples_;

    public:

        LabelSamplingWithoutReplacementConfig();

        uint32 getNumSamples() const override;

        ILabelSamplingWithoutReplacementConfig& setNumSamples(uint32 numSamples) override;

        std::unique_ptr<ILabelSamplingFactory> createLabelSamplingFactory(
          const ILabelMatrix& labelMatrix) const override;
};
