/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/indices/index_vector_complete.hpp"
#include "common/rule_refinement/prediction_evaluated.hpp"

/**
 * Stores the scores that are predicted by a rule that predicts for all available labels.
 */
class CompletePrediction final : public AbstractEvaluatedPrediction {
    private:

        const CompleteIndexVector indexVector_;

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        CompletePrediction(uint32 numElements);

        /**
         * An iterator that provides read-only access to the indices of the labels for which the rule predicts.
         */
        typedef CompleteIndexVector::const_iterator index_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices of the labels for which the rule predicts.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices of the labels for which the rule predicts.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

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
