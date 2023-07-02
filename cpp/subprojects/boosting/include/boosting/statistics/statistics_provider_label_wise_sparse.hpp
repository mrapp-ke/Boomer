/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss_label_wise_sparse.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_sparse.hpp"
#include "boosting/statistics/statistics_label_wise.hpp"
#include "common/statistics/statistics_provider.hpp"

namespace boosting {

    /**
     * Allows to create instances of the class `IStatisticsProvider` that provide access to an object of type
     * `ILabelWiseStatistics`, which uses sparse data structures to store the statistics.
     */
    class SparseLabelWiseStatisticsProviderFactory final : public IStatisticsProviderFactory {
        private:

            const std::unique_ptr<ISparseLabelWiseLossFactory> lossFactoryPtr_;

            const std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr_;

            const std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr_;

            const std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr_;

            const uint32 numThreads_;

        public:

            /**
             * @param lossFactoryPtr                    An unique pointer to an object of type
             *                                          `ISparseLabelWiseLossFactory` that allows to create
             *                                          implementations of the loss function that should be used for
             *                                          calculating gradients and Hessians
             * @param evaluationMeasureFactoryPtr       An unique pointer to an object of type
             *                                          `ISparseEvaluationMeasureFactory` that allows to create
             *                                          implementations of the evaluation measure that should be used
             *                                          for assessing the quality of predictions
             * @param regularRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `ISparseLabelWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of all remaining rules
             * @param pruningRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `ISparseLabelWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, when pruning rules
             * @param numThreads                        The number of CPU threads to be used to calculate the initial
             *                                          statistics in parallel. Must be at least 1
             */
            SparseLabelWiseStatisticsProviderFactory(
              std::unique_ptr<ISparseLabelWiseLossFactory> lossFactoryPtr,
              std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
              std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
              std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
              uint32 numThreads);

            /**
             * @see `IStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(
              const CContiguousConstView<const uint8>& labelMatrix) const override;

            /**
             * @see `IStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(const BinaryCsrConstView& labelMatrix) const override;
    };

}
