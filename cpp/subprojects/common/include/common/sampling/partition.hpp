/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <memory>

// Forward declarations
class IWeightVector;
class IInstanceSubSampling;
class RNG;
class IThresholdsSubset;
class ICoverageState;
class Refinement;
class AbstractPrediction;


/**
 * Defines an interface for all classes that provide access to the indices of training examples that have been split
 * into a training set and a holdout set.
 */
class IPartition {

    public:

        virtual ~IPartition() { };

        /**
         * Creates and returns a sub-sample of the examples that belong to the training set.
         *
         * @param instanceSubSampling   A reference to an object of type `IInstanceSubSampling` that should be used to
         *                              sample the examples
         * @param rng                   A reference to an object of type `RNG`, implementing the random number generator
         *                              to be used
         * @return                      An unique pointer to an object type `WeightVector` that provides access to the
         *                              weights of the individual training examples
         */
        virtual std::unique_ptr<IWeightVector> subSample(const IInstanceSubSampling& instanceSubSampling,
                                                         RNG& rng) const = 0;

        /**
         * Calculates and returns a quality score that assesses the quality of a rule's prediction for all examples that
         * do not belong to the current sub-sample and are marked as covered according to a given object of type
         * `ICoverageState`.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to
         *                          evaluate the prediction
         * @param coverageState     A reference to an object of type `ICoverageState` that keeps track of the examples
         *                          that are covered by the rule
         * @param head              A reference to an object of type `AbstractPrediction` that stores the scores that
         *                          are predicted by the rule
         * @return                  The calculated quality score
         */
        virtual float64 evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset,
                                            const ICoverageState& coverageState,
                                            const AbstractPrediction& head) = 0;

        /**
         * Recalculates the scores to be predicted by a refinement based on all examples in the training set that are
         * marked as covered according to a given object of type `ICoverageState` and updates the head of the refinement
         * accordingly.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to
         *                          recalculate the scores
         * @param coverageState     A reference to an object of type `ICoverageState` that keeps track of the examples
         *                          that are covered by the refinement
         * @param refinement        A reference to an object of type `Refinement`, whose head should be updated
         */
        virtual void recalculatePrediction(const IThresholdsSubset& thresholdsSubset,
                                           const ICoverageState& coverageState, Refinement& refinement) = 0;

};
