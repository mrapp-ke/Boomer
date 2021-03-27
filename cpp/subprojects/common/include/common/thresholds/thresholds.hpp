/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/thresholds/thresholds_subset.hpp"
#include "common/sampling/weight_vector.hpp"


/**
 * Defines an interface for all classes that provide access to thresholds that may be used by the first condition of a
 * rule that currently has an empty body and therefore covers the entire instance space.
 */
class IThresholds {

    public:

        virtual ~IThresholds() { };

        /**
         * Creates and returns a new subset of the thresholds, which initially contains all of the thresholds.
         *
         * @param weights   A reference to an object of type `IWeightVector` that provides access to the weights of the
         *                  individual training examples
         * @return          An unique pointer to an object of type `IThresholdsSubset` that has been created
         */
        virtual std::unique_ptr<IThresholdsSubset> createSubset(const IWeightVector& weights) = 0;

        /**
         * Returns the number of available examples.
         *
         * @return The number of examples
         */
        virtual uint32 getNumExamples() const = 0;

        /**
         * Returns the number of available features.
         *
         * @return The number of features
         */
        virtual uint32 getNumFeatures() const = 0;

        /**
         * Returns the number of available labels.
         *
         * @return The number of labels
         */
        virtual uint32 getNumLabels() const = 0;

};
