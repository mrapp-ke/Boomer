/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/instance_sampling.hpp"


/**
 * Allows to configure a method for sampling training examples that does not perform any sampling, but assigns equal
 * weights to all examples.
 */
class NoInstanceSamplingConfig final : public IInstanceSamplingConfig {

    public:

        std::unique_ptr<IInstanceSamplingFactory> createInstanceSamplingFactory() const override;

};
