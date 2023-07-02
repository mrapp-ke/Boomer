/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "boosting/losses/loss_label_wise.hpp"
#include "boosting/statistics/statistics_label_wise.hpp"
#include "common/statistics/statistics_provider.hpp"

namespace boosting {

    /**
     * Allows to create instances of the class `IStatisticsProvider` that provide access to an object of type
     * `ILabelWiseStatistics`, which uses dense data structures to store the statistics.
     */
    class DenseLabelWiseStatisticsProviderFactory final : public IStatisticsProviderFactory {
        private:

            const std::unique_ptr<ILabelWiseLossFactory> lossFactoryPtr_;

            const std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr_;

            const std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr_;

            const std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr_;

            const std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr_;

            const uint32 numThreads_;

        public:

            /**
             * @param lossFactoryPtr                    An unique pointer to an object of type `ILabelWiseLossFactory`
             *                                          that allows to create implementations of the loss function that
             *                                          should be used for calculating gradients and Hessians
             * @param evaluationMeasureFactoryPtr       An unique pointer to an object of type
             *                                          `IEvaluationMeasureFactory` that allows to create
             *                                          implementations of the evaluation measure that should be used
             *                                          for assessing the quality of predictions
             * @param defaultRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `ILabelWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of the default rule
             * @param regularRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `ILabelWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of all remaining rules
             * @param pruningRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `ILabelWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, when pruning rules
             * @param numThreads                        The number of CPU threads to be used to calculate the initial
             *                                          statistics in parallel. Must be at least 1
             */
            DenseLabelWiseStatisticsProviderFactory(
              std::unique_ptr<ILabelWiseLossFactory> lossFactoryPtr,
              std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
              std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
              std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
              std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads);

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
