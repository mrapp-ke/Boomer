/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/label_sampling.hpp"


/**
 * An implementation of the class `ILabelSubSampling` that does not perform any sampling, but includes all labels.
 */
class NoLabelSubSampling final : public ILabelSubSampling {

    public:

        std::unique_ptr<IIndexVector> subSample(uint32 numLabels, RNG& rng) const override;

};
