/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/weight_vector.hpp"
#include "common/sampling/random.hpp"
#include <memory>

// Forward declarations
class BiPartition;
class SinglePartition;


/**
 * Defines an interface for all classes that implement a strategy for sub-sampling training examples.
 */
class IInstanceSubSampling {

    public:

        virtual ~IInstanceSubSampling() { };

        /**
         * Creates and returns a sub-sample of the examples in a training set.
         *
         * @param partition A reference to an object of type `SinglePartition` that provides access to the indices of
         *                  the training examples that are included in the training set
         * @param rng       A reference to an object of type `RNG`, implementing the random number generator to be used
         * @return          An unique pointer to an object type `WeightVector` that provides access to the weights of
         *                  the individual training examples
         */
        virtual std::unique_ptr<IWeightVector> subSample(const SinglePartition& partition, RNG& rng) const = 0;

        /**
         * Creates and returns a sub-sample of the examples in a training set.
         *
         * @param partition An unique pointer to an object of type `BiPartition` that provides access to the indices of
         *                  the training examples that are included in the training set and the holdout set,
         *                  respectively
         * @param rng       A reference to an object of type `RNG`, implementing the random number generator to be used
         * @return          An unique pointer to an object type `WeightVector` that provides access to the weights of
         *                  the individual training examples
         */
        virtual std::unique_ptr<IWeightVector> subSample(const BiPartition& partition, RNG& rng) const = 0;

};
