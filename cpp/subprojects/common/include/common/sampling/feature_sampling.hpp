/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/indices/index_vector.hpp"
#include "common/input/feature_matrix.hpp"
#include "common/sampling/random.hpp"

#include <memory>

/**
 * Defines an interface for all classes that implement a method for sampling features.
 */
class IFeatureSampling {
    public:

        virtual ~IFeatureSampling() {};

        /**
         * Creates and returns a sample of the available features.
         *
         * @param rng   A reference to an object of type `RNG`, implementing the random number generator to be used
         * @return      A reference to an object of type `IIndexVector` that provides access to the indices of the
         *              features that are contained in the sample
         */
        virtual const IIndexVector& sample(RNG& rng) = 0;

        /**
         * Creates and returns a new object of type `IFeatureSampling` that is suited for use during a beam search.
         *
         * @param rng       A reference to an object of type `RNG`, implementing the random number generator be used
         * @param resample  True, if a new sample of the available features should be created whenever the sampling
         *                  method is invoked during the beam search, false otherwise
         * @return An unique pointer to an object of type `IFeatureSampling` that has been created
         */
        virtual std::unique_ptr<IFeatureSampling> createBeamSearchFeatureSampling(RNG& rng, bool resample) = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IFeatureSampling`.
 */
class IFeatureSamplingFactory {
    public:

        virtual ~IFeatureSamplingFactory() {};

        /**
         * Creates and returns a new object of type `IFeatureSampling`.
         *
         * @return An unique pointer to an object of type `IFeatureSampling` that has been created
         */
        virtual std::unique_ptr<IFeatureSampling> create() const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for sampling features.
 */
class IFeatureSamplingConfig {
    public:

        virtual ~IFeatureSamplingConfig() {};

        /**
         * Creates and returns a new object of type `IFeatureSamplingFactory` according to the specified configuration.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the features
         *                      of the training examples
         * @return              An unique pointer to an object of type `IFeatureSamplingFactory` that has been created
         */
        virtual std::unique_ptr<IFeatureSamplingFactory> createFeatureSamplingFactory(
          const IFeatureMatrix& featureMatrix) const = 0;

        /**
         * Returns whether feature sampling is used or not.
         *
         * @return True, if feature sampling is used, false otherwise
         */
        virtual bool isSamplingUsed() const = 0;
};
