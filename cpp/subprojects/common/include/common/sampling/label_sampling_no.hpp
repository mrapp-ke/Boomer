/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/label_sampling.hpp"


/**
 * Allows to create objects of the class `ILabelSampling` that do not perform any sampling, but include all labels.
 */
class NoLabelSamplingFactory final : public ILabelSamplingFactory {

    public:

        std::unique_ptr<ILabelSampling> create(uint32 numLabels) const override;

};
