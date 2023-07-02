/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/indices/index_vector_partial.hpp"
#include "common/rule_refinement/prediction_evaluated.hpp"

/**
 * Stores the scores that are predicted by a rule that predicts for a subset of the available labels.
 */
class PartialPrediction final : public AbstractEvaluatedPrediction {
    private:

        PartialIndexVector indexVector_;

        bool sorted_;

    public:

        /**
         * @param numElements   The number of labels for which the rule predicts
         * @param sorted        True, if the scores that are stored by this prediction are sorted in increasing order by
         *                      the corresponding label indices, false otherwise
         */
        PartialPrediction(uint32 numElements, bool sorted);

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

        /**
         * Sets the number of labels for which the rule predicts.
         *
         * @param numElements   The number of labels to be set
         * @param freeMemory    True, if unused memory should be freed if possible, false otherwise
         */
        void setNumElements(uint32 numElements, bool freeMemory);

        /**
         * Sets whether the scores that are stored by this prediction are sorted in increasing order by the
         * corresponding label indices, or not.
         *
         * @param sorted True, if the scores that are stored by this prediction are sorted in increasing order by the
         *               corresponding label indices, false otherwise
         */
        void setSorted(bool sorted);

        bool isPartial() const override;

        uint32 getIndex(uint32 pos) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(const IStatistics& statistics,
                                                                  const EqualWeightVector& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(const IStatistics& statistics,
                                                                  const BitWeightVector& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const DenseWeightVector<uint32>& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const OutOfSampleWeightVector<EqualWeightVector>& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const OutOfSampleWeightVector<BitWeightVector>& weights) const override;

        std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics,
          const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override;

        std::unique_ptr<IRuleRefinement> createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                              uint32 featureIndex) const override;

        void apply(IStatistics& statistics, uint32 statisticIndex) const override;

        void revert(IStatistics& statistics, uint32 statisticIndex) const override;

        void sort() override;

        std::unique_ptr<IHead> createHead() const override;
};
