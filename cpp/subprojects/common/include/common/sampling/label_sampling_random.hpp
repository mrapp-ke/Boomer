/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/label_sampling.hpp"


/**
 * Implements random label subset selection for selecting a random subset of the available features without replacement.
 */
class RandomLabelSubsetSelection final : public ILabelSubSampling {

    private:

        uint32 numSamples_;

    public:

        /**
         * @param numSamples The number of labels to be included in the sample
         */
        RandomLabelSubsetSelection(uint32 numSamples);

        std::unique_ptr<IIndexVector> subSample(uint32 numLabels, RNG& rng) const override;

};
