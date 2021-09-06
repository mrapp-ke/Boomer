/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/thresholds/coverage_mask.hpp"
#include "common/thresholds/coverage_set.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/rule_refinement/rule_refinement.hpp"
#include "common/rule_refinement/prediction.hpp"
#include "common/model/condition.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include <memory>


/**
 * Defines an interface for all classes that provide access a subset of thresholds that may be used by the conditions of
 * a rule with arbitrary body. The thresholds may include only those that correspond to the subspace of the instance
 * space that is covered by the rule.
 */
class IThresholdsSubset {

    public:

        virtual ~IThresholdsSubset() { };

        /**
         * Creates and returns a new instance of the type `IRuleRefinement` that allows to find the best refinement of
         * an existing rule that predicts for all available labels.
         *
         * @param labelIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the labels for which the existing rule predicts
         * @param featureIndex  The index of the feature that should be considered when searching for refinements
         * @return              An unique pointer to an object of type `IRuleRefinement` that has been created
         */
        virtual std::unique_ptr<IRuleRefinement> createRuleRefinement(const CompleteIndexVector& labelIndices,
                                                                      uint32 featureIndex) = 0;

        /**
         * Creates and returns a new instance of the type `IRuleRefinement` that allows to find the best refinement of
         * an existing rule that predicts for a subset of the available labels.
         *
         * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the labels for which the existing rule predicts
         * @param featureIndex  The index of the feature that should be considered when searching for refinements
         * @return              An unique pointer to an object of type `IRuleRefinement` that has been created
         */
        virtual std::unique_ptr<IRuleRefinement> createRuleRefinement(const PartialIndexVector& labelIndices,
                                                                      uint32 featureIndex) = 0;

        /**
         * Filters the thresholds such that only those thresholds, which correspond to the instance space that is
         * covered by a specific refinement of a rule, are included.
         *
         * The given refinement must have been found by an instance of type `IRuleRefinement` that was previously
         * created via the function `createRuleRefinement`. The function function `resetThresholds` must not have been
         * called since.
         *
         * @param refinement A reference to an object of type `Refinement` that stores information about the refinement
         */
        virtual void filterThresholds(Refinement& refinement) = 0;

        /**
         * Filters the thresholds such that only those thresholds, which correspond to the instance space that is
         * covered by specific condition of a rule, are included.
         *
         * Unlike the function `filterThresholds(Refinement)`, the given condition must not have been found by an
         * instance of `IRuleRefinement` and the function `resetThresholds` may have been called before.
         *
         * @param condition A reference to an object of type `Refinement` that stores information about the condition
         */
        virtual void filterThresholds(const Condition& condition) = 0;

        /**
         * Resets the filtered thresholds. This reverts the effects of all previous calls to the functions
         * `filterThresholds(Refinement)` or `filterThresholds(Condition)`.
         */
        virtual void resetThresholds() = 0;

        /**
         * Returns an object of type `ICoverageState` that keeps track of the elements that are covered by the
         * refinement that has been applied via the function `applyRefinement`.
         *
         * @return A reference to an object of type `ICoverageState` that keeps track of the elements that are covered
         *         by the refinement
         */
        virtual const ICoverageState& getCoverageState() const = 0;

        /**
         * Calculates and returns a quality score that assesses the quality of a rule's prediction for all examples that
         * do not belong to the current sub-sample and are marked as covered according to a given object of type
         * `CoverageMask`.
         *
         * For calculating the quality score, only examples that belong to the training set and are not included in the
         * current sub-sample, i.e., only examples with zero weights, are considered.
         *
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that belong to the training set
         * @param coverageState A reference to an object of type `CoverageMask` that keeps track of the examples that
         *                      are covered by the rule
         * @param head          A reference to an object of type `AbstractPrediction` that stores the scores that are
         *                      predicted by the rule
         * @return              The calculated quality score
         */
        virtual float64 evaluateOutOfSample(const SinglePartition& partition, const CoverageMask& coverageState,
                                            const AbstractPrediction& head) const = 0;


        /**
         * Calculates and returns a quality score that assesses the quality of a rule's prediction for all examples that
         * do not belong to the current sub-sample and are marked as covered according to a given object of type
         * `CoverageMask`.
         *
         * For calculating the quality score, only examples that belong to the training set and are not included in the
         * current sub-sample, i.e., only examples with zero weights, are considered.
         *
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that belong to the training set
         * @param coverageState A reference to an object of type `CoverageMask` that keeps track of the examples that
         *                      are covered by the rule
         * @param head          A reference to an object of type `AbstractPrediction` that stores the scores that are
         *                      predicted by the rule
         * @return              The calculated quality score
         */
        virtual float64 evaluateOutOfSample(const BiPartition& partition, const CoverageMask& coverageState,
                                            const AbstractPrediction& head) const = 0;

        /**
         * Calculates and returns a quality score that assesses the quality of a rule's prediction for all examples that
         * do not belong to the current sub-sample and are marked as covered according to a given object of type
         * `CoverageSet`.
         *
         * For calculating the quality score, only examples that belong to the training set and are not included in the
         * current sub-sample, i.e., only examples with zero weights, are considered.
         *
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that belong to the training set
         * @param coverageState A reference to an object of type `CoverageSet` that keeps track of the examples that are
         *                      covered by the rule
         * @param head          A reference to an object of type `AbstractPrediction` that stores the scores that are
         *                      predicted by the rule
         * @return              The calculated quality score
         */
        virtual float64 evaluateOutOfSample(const SinglePartition& partition, const CoverageSet& coverageState,
                                            const AbstractPrediction& head) const = 0;


        /**
         * Calculates and returns a quality score that assesses the quality of a rule's prediction for all examples that
         * do not belong to the current sub-sample and are marked as covered according to a given object of type
         * `CoverageSet`.
         *
         * For calculating the quality score, only examples that belong to the training set and are not included in the
         * current sub-sample, i.e., only examples with zero weights, are considered.
         *
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that belong to the training set
         * @param coverageState A reference to an object of type `Coverageset` that keeps track of the examples that are
         *                      covered by the rule
         * @param head          A reference to an object of type `AbstractPrediction` that stores the scores that are
         *                      predicted by the rule
         * @return              The calculated quality score
         */
        virtual float64 evaluateOutOfSample(BiPartition& partition, const CoverageSet& coverageState,
                                            const AbstractPrediction& head) const = 0;

        /**
         * Recalculates the scores to be predicted by a refinement based on all examples in the training set that are
         * marked as covered according to a given object of type `CoverageMask` and updates the head of the refinement
         * accordingly.
         *
         * When calculating the updated scores, the weights of the individual training examples are ignored and equally
         * distributed weights are assumed instead.
         *
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that belong to the training set
         * @param coverageState A reference to an object of type `CoverageMask` that keeps track of the examples that
         *                      are covered by the refinement
         * @param refinement    A reference to an object of type `Refinement`, whose head should be updated
         */
        virtual void recalculatePrediction(const SinglePartition& partition, const CoverageMask& coverageState,
                                           Refinement& refinement) const = 0;

        /**
         * Recalculates the scores to be predicted by a refinement based on all examples in the training set that are
         * marked as covered according to a given object of type `CoverageMask` and updates the head of the refinement
         * accordingly.
         *
         * When calculating the updated scores, the weights of the individual training examples are ignored and equally
         * distributed weights are assumed instead.
         *
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that belong to the training set
         * @param coverageState A reference to an object of type `CoverageMask` that keeps track of the examples that
         *                      are covered by the refinement
         * @param refinement    A reference to an object of type `Refinement`, whose head should be updated
         */
        virtual void recalculatePrediction(const BiPartition& partition, const CoverageMask& coverageState,
                                           Refinement& refinement) const = 0;

        /**
         * Recalculates the scores to be predicted by a refinement based on all examples in the training set that are
         * marked as covered according to a given object of type `CoverageSet` and updates the head of the refinement
         * accordingly.
         *
         * When calculating the updated scores, the weights of the individual training examples are ignored and equally
         * distributed weights are assumed instead.
         *
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that belong to the training set
         * @param coverageState A reference to an object of type `CoverageMask` that keeps track of the examples that
         *                      are covered by the refinement
         * @param refinement    A reference to an object of type `Refinement`, whose head should be updated
         */
        virtual void recalculatePrediction(const SinglePartition& partition, const CoverageSet& coverageState,
                                           Refinement& refinement) const = 0;

        /**
         * Recalculates the scores to be predicted by a refinement based on all examples in the training set that are
         * marked as covered according to a given object of type `CoverageSet` and updates the head of the refinement
         * accordingly.
         *
         * When calculating the updated scores, the weights of the individual training examples are ignored and equally
         * distributed weights are assumed instead.
         *
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that belong to the training set
         * @param coverageState A reference to an object of type `CoverageSet` that keeps track of the examples that are
         *                      covered by the refinement
         * @param refinement    A reference to an object of type `Refinement`, whose head should be updated
         */
        virtual void recalculatePrediction(BiPartition& partition, const CoverageSet& coverageState,
                                           Refinement& refinement) const = 0;

        /**
         * Applies the predictions of a rule to the statistics that correspond to the current subset.
         *
         * @param prediction A reference to an object of type `AbstractPrediction`, representing the predictions to be
         *                   applied
         */
        virtual void applyPrediction(const AbstractPrediction& prediction) = 0;

};
