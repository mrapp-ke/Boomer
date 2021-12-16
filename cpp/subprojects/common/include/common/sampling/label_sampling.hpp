/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/indices/index_vector.hpp"
#include "common/sampling/random.hpp"
#include <memory>


/**
 * Defines an interface for all classes that implement a strategy for sampling labels.
 */
class ILabelSampling {

    public:

        virtual ~ILabelSampling() { };

        /**
         * Creates and returns a sample of the available labels.
         *
         * @param rng   A reference to an object of type `RNG`, implementing the random number generator to be used
         * @return      A reference to an object of type `IIndexVector` that provides access to the indices of the
         *              labels that are contained in the sample
         */
        virtual const IIndexVector& sample(RNG& rng) = 0;

};

/**
 * Defines an interface for all factories that allow to create objects of type `ILabelSampling`.
 */
class ILabelSamplingFactory {

    public:

        virtual ~ILabelSamplingFactory() { };

        /**
         * Creates and returns a new object of type `ILabelSampling`.
         *
         * @param numLabels The total number of available labels
         * @return          An unique pointer to an object of type `ILabelSampling` that has been created
         */
        virtual std::unique_ptr<ILabelSampling> create(uint32 numLabels) const = 0;

};
