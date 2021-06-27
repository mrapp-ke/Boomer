/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/partition_sampling.hpp"


/**
 * An implementation of the class `IPartitionSampling` that does not split the training examples, but includes all of
 * them in the training set.
 */
class NoPartitionSampling : public IPartitionSampling {

    public:

        std::unique_ptr<IPartition> partition(uint32 numExamples, RNG& rng) const override;

};
