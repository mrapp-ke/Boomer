/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"

#include <memory>

// Forward declarations
class IThresholds;
class IThresholdsSubset;

/**
 * Defines an interface for one-dimensional vectors that provide access to weights.
 */
class IWeightVector {
    public:

        virtual ~IWeightVector() {};

        /**
         * Returns whether the vector contains any zero weights or not.
         *
         * @return True, if the vector contains any zero weights, false otherwise
         */
        virtual bool hasZeroWeights() const = 0;

        /**
         * Creates and returns a new instance of type `IThresholdsSubset` that provides access to the statistics that
         * correspond to individual training examples whose weights are stored in this vector.
         *
         * @param thresholds    A reference to an object of type `IThresholds` that should be used to create the
         *                      instance
         * @return              An unique pointer to an object of type `IThresholdsSubset` that has been created
         */
        virtual std::unique_ptr<IThresholdsSubset> createThresholdsSubset(IThresholds& thresholds) const = 0;
};
