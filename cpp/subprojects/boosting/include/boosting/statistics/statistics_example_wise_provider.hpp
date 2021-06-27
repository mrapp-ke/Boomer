/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics_provider_factory.hpp"
#include "boosting/losses/loss_example_wise.hpp"
#include "boosting/statistics/statistics_example_wise.hpp"


namespace boosting {

    /**
     * Allows to create instances of the class `ExampleWiseStatisticsProvider`.
     */
    class ExampleWiseStatisticsProviderFactory: public IStatisticsProviderFactory {

        private:

            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr_;

            std::shared_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr_;

            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param lossFunctionPtr                   A shared pointer to an object of type `IExampleWiseLoss` that
             *                                          should be used for calculating gradients and Hessians
             * @param defaultRuleEvaluationFactoryPtr   A shared pointer to an object of type
             *                                          `IExampleWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of the default rule
             * @param ruleEvaluationFactoryPtr          A shared pointer to an object of type
             *                                          `IExampleWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of all remaining rules
             * @param numThreads                        The number of CPU threads to be used to calculate the initial
             *                                          statistics in parallel. Must be at least 1
             */
            ExampleWiseStatisticsProviderFactory(
                std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
                std::shared_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
                std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, uint32 numThreads);

            std::unique_ptr<IStatisticsProvider> create(
                std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr) const override;

    };

}
