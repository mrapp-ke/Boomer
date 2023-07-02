/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/multi_threading/multi_threading.hpp"

/**
 * Allows to configure the multi-threading behavior of a parallelize algorithm that should not use any multi-threading.
 */
class NoMultiThreadingConfig final : public IMultiThreadingConfig {
    public:

        uint32 getNumThreads(const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;
};
