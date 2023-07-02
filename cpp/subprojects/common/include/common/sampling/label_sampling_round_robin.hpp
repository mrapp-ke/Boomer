/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/macros.hpp"
#include "common/sampling/label_sampling.hpp"

/**
 * Allows to configure a method for sampling labels in a round-robin fashion.
 */
class RoundRobinLabelSamplingConfig final : public ILabelSamplingConfig {
    public:

        std::unique_ptr<ILabelSamplingFactory> createLabelSamplingFactory(
          const ILabelMatrix& labelMatrix) const override;
};
