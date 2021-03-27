/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/partition.hpp"
#include "common/sampling/random.hpp"
#include <memory>


/**
 * Defines an interface for all classes that implement a strategy for partitioning the available training examples into
 * a training set and a holdout set.
 */
class IPartitionSampling {

    public:

        virtual ~IPartitionSampling() { };

        /**
         * Creates and returns a partition of the available training examples.
         *
         * @param numExamples   The total number of available training examples
         * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be
         *                      used
         * @return              An unique pointer to an object of type `IPartition` that provides access to the indices
         *                      of the training examples that belong to the training set and holdout set, respectively
         */
        virtual std::unique_ptr<IPartition> partition(uint32 numExamples, RNG& rng) const = 0;

};
