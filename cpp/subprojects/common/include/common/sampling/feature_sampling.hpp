/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/indices/index_vector.hpp"
#include "common/sampling/random.hpp"
#include <memory>


/**
 * Defines an interface for all classes that implement a strategy for sub-sampling features.
 */
class IFeatureSubSampling {

    public:

        virtual ~IFeatureSubSampling() { };

        /**
         * Creates and returns a sub-sample of the available features.
         *
         * @param numFeatures   The total number of available features
         * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be
         *                      used
         * @return              An unique pointer to an object of type `IIndexVector` that provides access to the
         *                      indices of the features that are contained in the sub-sample
         */
        virtual std::unique_ptr<IIndexVector> subSample(uint32 numFeatures, RNG& rng) const = 0;

};
