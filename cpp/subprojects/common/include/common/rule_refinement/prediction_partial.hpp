/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_refinement/prediction_evaluated.hpp"
#include "common/indices/index_vector_partial.hpp"

// Forward declarations
class IImmutableStatistics;


/**
 * Stores the scores that are predicted by a rule that predicts for a subset of the available labels.
 */
class PartialPrediction final : public AbstractEvaluatedPrediction {

    private:

        PartialIndexVector indexVector_;

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        PartialPrediction(uint32 numElements);

        /**
         * An iterator that provides access to the indices for which the rule predicts and allows to modify them.
         */
        typedef PartialIndexVector::iterator index_iterator;

        /**
         * An iterator that provides read-only access to the indices for which the rule predicts.
         */
        typedef PartialIndexVector::const_iterator index_const_iterator;

        /**
         * Returns an `index_iterator` to the beginning of the indices for which the rule predicts.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indices_begin();

        /**
         * Returns an `index_iterator` to the end of the indices for which the rule predicts.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the indices for which the rule predicts.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices for which the rule predicts.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        void setNumElements(uint32 numElements, bool freeMemory) override;

        bool isPartial() const override;

        uint32 getIndex(uint32 pos) const override;

        std::unique_ptr<IStatisticsSubset> createSubset(const IImmutableStatistics& statistics) const override;

        std::unique_ptr<IRuleRefinement> createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                              uint32 featureIndex) const override;

        void apply(IStatistics& statistics, uint32 statisticIndex) const override;

        std::unique_ptr<IHead> toHead() const override;

};
