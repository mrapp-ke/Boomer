/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <memory>

// Forward declarations
class IThresholdsSubset;
class SinglePartition;
class BiPartition;
class AbstractPrediction;
class Refinement;


/**
 * Defines an interface for all classes that allow to keep track of the examples that are covered by a rule.
 */
class ICoverageState {

    public:

        virtual ~ICoverageState() { };

        /**
         * Creates and returns a deep copy of the coverage state.
         *
         * @return An unique pointer to an object of type `ICoverageState` that has been created
         */
        virtual std::unique_ptr<ICoverageState> copy() const = 0;

        /**
         * Calculates and returns a quality score that assesses the quality of a rule's prediction for all examples that
         * do not belong to the current sub-sample and are marked as covered.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to
         *                          evaluate the prediction
         * @param partition         A reference to an object of type `SinglePartition` that provides access to the
         *                          indices of the training examples that belong to the training set
         * @param head              A reference to an object of type `AbstractPrediction` that stores the scores that
         *                          are predicted by the rule
         * @return                  The calculated quality score
         */
        virtual float64 evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                            const AbstractPrediction& head) const = 0;

        /**
         * Calculates and returns a quality score that assesses the quality of a rule's prediction for all examples that
         * do not belong to the current sub-sample and are marked as covered.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to
         *                          evaluate the prediction
         * @param partition         A reference to an object of type `BiPartition` that provides access to the indices
         *                          of the training examples that belong to the training set
         * @param head              A reference to an object of type `AbstractPrediction` that stores the scores that
         *                          are predicted by the rule
         * @return                  The calculated quality score
         */
        virtual float64 evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                            const AbstractPrediction& head) const = 0;

        /**
         * Recalculates the scores to be predicted by a refinement based on all examples in the training set that are
         * marked as covered and updates the head of the refinement accordingly.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to
         *                          recalculate the scores
         * @param partition         A reference to an object of type `SinglePartition` that provides access to the
         *                          indices of the training examples that belong to the training set
         * @param refinement        A reference to an object of type `Refinement`, whose head should be updated
         */
        virtual void recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                           Refinement& refinement) const = 0;

        /**
         * Recalculates the scores to be predicted by a refinement based on all examples in the training set that are
         * marked as covered and updates the head of the refinement accordingly.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to
         *                          recalculate the scores
         * @param partition         A reference to an object of type `BiPartition` that provides access to the indices
         *                          of the training examples that belong to the training set
         * @param refinement        A reference to an object of type `Refinement`, whose head should be updated
         */
        virtual void recalculatePrediction(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                           Refinement& refinement) const = 0;

};
