/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include <memory>


/**
 * Defines an interface for all classes that allow to transform the scores that are predicted for individual labels into
 * probabilities.
 */
class IProbabilityFunction {

    public:

        virtual ~IProbabilityFunction() { };

        /**
         * Transforms the score that is predicted for an individual label into a probability.
         *
         * @param predictedScore    The predicted score
         * @return                  The probability
         */
        virtual float64 transform(float64 predictedScore) const = 0;

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IProbabilityFunction`.
 */
class IProbabilityFunctionFactory {

    public:

        virtual ~IProbabilityFunctionFactory() { };

        /**
         * Creates and returns a new object of the type `IProbabilityFunction`.
         *
         * @return An unique pointer to an object of type `IProbabilityFunction` that has been created
         */
        virtual std::unique_ptr<IProbabilityFunction> create() const = 0;

};
