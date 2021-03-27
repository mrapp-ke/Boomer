/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/partition_sampling.hpp"


/**
 * An implementation of the class `IPartitionSampling` that splits the training examples into two mutually exclusive
 * sets that may be used as a training set and a holdout set.
 */
class BiPartitionSampling : public IPartitionSampling {

    private:

        float32 holdoutSetSize_;

    public:

        /**
         * @param holdoutSetSize The fraction of examples to be included in the holdout set (e.g. a value of 0.6
         *                       corresponds to 60 % of the available examples). Must be in (0, 1)
         */
        BiPartitionSampling(float32 holdoutSetSize);

        std::unique_ptr<IPartition> partition(uint32 numExamples, RNG& rng) const override;

};
