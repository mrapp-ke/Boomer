/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/indices/index_vector.hpp"
#include "common/sampling/random.hpp"
#include <memory>


/**
 * Defines an interface for all classes that implement a strategy for sampling features.
 */
class IFeatureSampling {

    public:

        virtual ~IFeatureSampling() { };

        /**
         * Creates and returns a sample of the available features.
         *
         * @param rng   A reference to an object of type `RNG`, implementing the random number generator to be used
         * @return      A reference to an object of type `IIndexVector` that provides access to the indices of the
         *              features that are contained in the sample
         */
        virtual const IIndexVector& sample(RNG& rng) = 0;

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IFeatureSampling`.
 */
class IFeatureSamplingFactory {

    public:

        virtual ~IFeatureSamplingFactory() { };

        /**
         * Creates and returns a new object of type `IFeatureSampling`.
         *
         * @param numFeatures   The total number of available features
         * @return              An unique pointer to an object of type `IFeatureSampling` that has been created
         */
        virtual std::unique_ptr<IFeatureSampling> create(uint32 numFeatures) const = 0;

};
