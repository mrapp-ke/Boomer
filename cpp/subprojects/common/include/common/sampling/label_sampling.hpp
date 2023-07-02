/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/indices/index_vector.hpp"
#include "common/input/label_matrix.hpp"
#include "common/sampling/random.hpp"

#include <memory>

/**
 * Defines an interface for all classes that implement a method for sampling labels.
 */
class ILabelSampling {
    public:

        virtual ~ILabelSampling() {};

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

        virtual ~ILabelSamplingFactory() {};

        /**
         * Creates and returns a new object of type `ILabelSampling`.
         *
         * @return An unique pointer to an object of type `ILabelSampling` that has been created
         */
        virtual std::unique_ptr<ILabelSampling> create() const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for sampling labels.
 */
class ILabelSamplingConfig {
    public:

        virtual ~ILabelSamplingConfig() {};

        /**
         * Creates and returns a new object of type `ILabelSamplingFactory` according to the specified configuration.
         *
         * @param labelMatrix   A reference to an object of type `ILabelMatrix` that provides access to the labels of
         *                      the training examples
         * @return              An unique pointer to an object of type `ILabelSamplingFactory` that has been created
         */
        virtual std::unique_ptr<ILabelSamplingFactory> createLabelSamplingFactory(
          const ILabelMatrix& labelMatrix) const = 0;
};
