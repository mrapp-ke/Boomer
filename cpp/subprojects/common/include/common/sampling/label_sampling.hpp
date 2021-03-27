/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/indices/index_vector.hpp"
#include "common/sampling/random.hpp"
#include <memory>


/**
 * Defines an interface for all classes that implement a strategy for sub-sampling labels.
 */
class ILabelSubSampling {

    public:

        virtual ~ILabelSubSampling() { };

        /**
         * Creates and returns a sub-sample of the available labels.
         *
         * @param numLabels The total number of available labels
         * @param rng       A reference to an object of type `RNG`, implementing the random number generator to be used
         * @return          An unique pointer to an object of type `IIndexVector` that provides access to the indices of
         *                  the labels that are contained in the sub-sample
         */
        virtual std::unique_ptr<IIndexVector> subSample(uint32 numLabels, RNG& rng) const = 0;

};
