/*
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/binning/threshold_vector.hpp"
#include "common/rule_refinement/rule_refinement.hpp"
#include "common/rule_refinement/rule_refinement_callback.hpp"
#include "common/statistics/histogram.hpp"

/**
 * Allows to find the best refinements of existing rules, which result from adding a new condition that correspond to a
 * certain feature. The thresholds that may be used by the new condition result from the boundaries between the bins
 * that have been created using a binning method.
 *
 * @tparam IndexVector The type of the vector that provides access to the indices of the labels for which the refined
 *                     rule is allowed to predict
 */
template<typename IndexVector>
class ApproximateRuleRefinement final : public IRuleRefinement {
    private:

        const IndexVector& labelIndices_;

        const uint32 numExamples_;

        const uint32 featureIndex_;

        const bool nominal_;

        typedef IRuleRefinementCallback<IHistogram, ThresholdVector> Callback;

        const std::unique_ptr<Callback> callbackPtr_;

    public:

        /**
         * @param labelIndices      A reference to an object of template type `IndexVector` that provides access to the
         *                          indices of the labels for which the refined rule is allowed to predict
         * @param numExamples       The total number of training examples with non-zero weights that are covered by the
         *                          existing rule
         * @param featureIndex      The index of the feature, the new condition corresponds to
         * @param nominal           True, if the feature at index `featureIndex` is nominal, false otherwise
         * @param callbackPtr       An unique pointer to an object of type `IRuleRefinementCallback` that allows to
         *                          retrieve the information that is required to search for potential refinements
         */
        ApproximateRuleRefinement(const IndexVector& labelIndices, uint32 numExamples, uint32 featureIndex,
                                  bool nominal, std::unique_ptr<Callback> callbackPtr);

        void findRefinement(SingleRefinementComparator& comparator, uint32 minCoverage) override;

        void findRefinement(FixedRefinementComparator& comparator, uint32 minCoverage) override;
};
