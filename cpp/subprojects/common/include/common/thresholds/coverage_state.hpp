/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/util/quality.hpp"

#include <memory>

// Forward declarations
class IThresholdsSubset;
class SinglePartition;
class BiPartition;
class AbstractPrediction;

/**
 * Defines an interface for all classes that allow to keep track of the examples that are covered by a rule.
 */
class ICoverageState {
    public:

        virtual ~ICoverageState() {};

        /**
         * Creates and returns a deep copy of the coverage state.
         *
         * @return An unique pointer to an object of type `ICoverageState` that has been created
         */
        virtual std::unique_ptr<ICoverageState> copy() const = 0;

        /**
         * Calculates and returns a numerical score that assesses the quality of a rule's prediction for all examples
         * that do not belong to the current sub-sample and are marked as covered.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to
         *                          evaluate the prediction
         * @param partition         A reference to an object of type `SinglePartition` that provides access to the
         *                          indices of the training examples that belong to the training set
         * @param head              A reference to an object of type `AbstractPrediction` that stores the scores that
         *                          are predicted by the rule
         * @return                  An object of type `Quality` that stores the calculated quality
         */
        virtual Quality evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                            const AbstractPrediction& head) const = 0;

        /**
         * Calculates and returns a numerical score that assesses the quality of a rule's prediction for all examples
         * that do not belong to the current sub-sample and are marked as covered.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to
         *                          evaluate the prediction
         * @param partition         A reference to an object of type `BiPartition` that provides access to the indices
         *                          of the training examples that belong to the training set
         * @param head              A reference to an object of type `AbstractPrediction` that stores the scores that
         *                          are predicted by the rule
         * @return                  An object of type `Quality` that stores the calculated quality
         */
        virtual Quality evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                            const AbstractPrediction& head) const = 0;

        /**
         * Recalculates and updates a rule's prediction based on all examples in the training set that are marked as
         * covered.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to
         *                          recalculate the prediction
         * @param partition         A reference to an object of type `SinglePartition` that provides access to the
         *                          indices of the training examples that belong to the training set
         * @param head              A reference to an object of type `AbstractPrediction` to be updated
         */
        virtual void recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                           AbstractPrediction& head) const = 0;

        /**
         * Recalculates and updates a rule's prediction based on all examples in the training set that are marked as
         * covered.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to
         *                          recalculate the prediction
         * @param partition         A reference to an object of type `BiPartition` that provides access to the indices
         *                          of the training examples that belong to the training set
         * @param head              A reference to an object of type `AbstractPrediction` to be updated
         */
        virtual void recalculatePrediction(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                           AbstractPrediction& head) const = 0;
};
