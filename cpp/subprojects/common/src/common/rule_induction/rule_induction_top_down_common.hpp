/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/thresholds/thresholds_subset.hpp"
#include "omp.h"

/**
 * Stores an unique pointer to an object of type `IRuleRefinement` that may be used to search for potential refinements
 * of a rule, as well as to an object of template type `RefinementComparator` that allows comparing different
 * refinements and keeping track of the best one(s).
 *
 * @tparam The type of the comparator that allows comparing different refinements and keeping track of the best one(s)
 */
template<typename RefinementComparator>
struct RuleRefinement final {
    public:

        /**
         * An unique pointer to an object of type `IRuleRefinement` that may be used to search for potential refinements
         * of a rule.
         */
        std::unique_ptr<IRuleRefinement> ruleRefinementPtr;

        /**
         * An unique pointer to an object of template type `RefinementComparator` that allows comparing different
         * refinements and keeping track of the best one(s).
         */
        std::unique_ptr<RefinementComparator> comparatorPtr;
};

/**
 * Finds the best refinement(s) of an existing rule across multiple features.
 *
 * @tparam RefinementComparator The type of the comparator that is used to compare the potential refinements
 * @param refinementComparator  A reference to an object of template type `RefinementComparator` that should be used to
 *                              compare the potential refinements
 * @param thresholdsSubset      A reference to an object of type `IThresholdsSubset` that should be used to search for
 *                              the potential refinements
 * @param featureIndices        A reference to an object of type `IIndexVector` that provides access to the indices of
 *                              the features that should be considered
 * @param labelIndices          A reference to an object of type `IIndexVector` that provides access to the indices of
 *                              the labels for which the refinement(s) may predict
 * @param minCoverage           The minimum number of training examples that must be covered by potential refinements
 * @param numThreads            The number of CPU threads to be used to search for potential refinements across multiple
 *                              features in parallel
 * @return                      True, if at least one refinement has been found, false otherwise
 */
template<typename RefinementComparator>
static inline bool findRefinement(RefinementComparator& refinementComparator, IThresholdsSubset& thresholdsSubset,
                                  const IIndexVector& featureIndices, const IIndexVector& labelIndices,
                                  uint32 minCoverage, uint32 numThreads) {
    bool foundRefinement = false;

    // For each feature, create an object of type `RuleRefinement<RefinementComparator>`...
    uint32 numFeatures = featureIndices.getNumElements();
    RuleRefinement<RefinementComparator>* ruleRefinements = new RuleRefinement<RefinementComparator>[numFeatures];

    for (uint32 i = 0; i < numFeatures; i++) {
        uint32 featureIndex = featureIndices.getIndex(i);
        RuleRefinement<RefinementComparator>& ruleRefinement = ruleRefinements[i];
        ruleRefinement.comparatorPtr = std::make_unique<RefinementComparator>(refinementComparator);
        ruleRefinement.ruleRefinementPtr = labelIndices.createRuleRefinement(thresholdsSubset, featureIndex);
    }

    // Search for the best condition among all available features to be added to the current rule...
#pragma omp parallel for firstprivate(numFeatures) firstprivate(ruleRefinements) firstprivate(minCoverage) \
  schedule(dynamic) num_threads(numThreads)
    for (int64 i = 0; i < numFeatures; i++) {
        RuleRefinement<RefinementComparator>& ruleRefinement = ruleRefinements[i];
        ruleRefinement.ruleRefinementPtr->findRefinement(*ruleRefinement.comparatorPtr, minCoverage);
    }

    // Pick the best refinement among the refinements that have been found for the different features...
    for (uint32 i = 0; i < numFeatures; i++) {
        RuleRefinement<RefinementComparator>& ruleRefinement = ruleRefinements[i];
        foundRefinement |= refinementComparator.merge(*ruleRefinement.comparatorPtr);
    }

    delete[] ruleRefinements;
    return foundRefinement;
}
