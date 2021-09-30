/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/label_sampling.hpp"


/**
 * Allows to create objects of type `ILabelSampling` that select a random subset of the available features without
 * replacement.
 */
class LabelSamplingWithoutReplacementFactory final : public ILabelSamplingFactory {

    private:

        uint32 numSamples_;

    public:

        /**
         * @param numSamples The number of labels to be included in the sample. Must be at least 1
         */
        LabelSamplingWithoutReplacementFactory(uint32 numSamples);

        std::unique_ptr<ILabelSampling> create(uint32 numLabels) const override;

};
