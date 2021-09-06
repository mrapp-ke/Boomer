/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics_provider_factory.hpp"
#include "boosting/statistics/statistics_label_wise.hpp"
#include "boosting/losses/loss_label_wise.hpp"


namespace boosting {

    /**
     * Allows to create instances of the class `IStatisticsProvider` that provide access to an object of type
     * `ILabelWiseStatistics`, which uses dense data structures to store the statistics.
     */
    class DenseLabelWiseStatisticsProviderFactory : public IStatisticsProviderFactory {

        private:

            std::unique_ptr<ILabelWiseLoss> lossFunctionPtr_;

            std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr_;

            std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr_;

            std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param lossFunctionPtr                   An unique pointer to an object of type `ILabelWiseLoss` that
             *                                          should be used for calculating gradients and Hessians
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
                std::unique_ptr<ILabelWiseLoss> lossFunctionPtr,
                std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
                std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
                std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads);

            std::unique_ptr<IStatisticsProvider> create(const CContiguousLabelMatrix& labelMatrix) const override;

            std::unique_ptr<IStatisticsProvider> create(const CsrLabelMatrix& labelMatrix) const override;

    };

}
