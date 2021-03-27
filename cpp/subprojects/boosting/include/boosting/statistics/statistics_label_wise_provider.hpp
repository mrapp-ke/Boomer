/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics_provider_factory.hpp"
#include "boosting/losses/loss_label_wise.hpp"
#include "boosting/statistics/statistics_label_wise.hpp"


namespace boosting {

    /**
     * Allows to create instances of the class `LabelWiseStatisticsProvider`.
     */
    class LabelWiseStatisticsProviderFactory : public IStatisticsProviderFactory {

        private:

            std::shared_ptr<ILabelWiseLoss> lossFunctionPtr_;

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr_;

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param lossFunctionPtr                   A shared pointer to an object of type `ILabelWiseLoss` that
             *                                          should be used for calculating gradients and Hessians
             * @param defaultRuleEvaluationFactoryPtr   A shared pointer to an object of type
             *                                          `ILabelWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of the default rule
             * @param ruleEvaluationFactoryPtr          A shared pointer to an object of type
             *                                          `ILabelWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of all remaining rules
             * @param numThreads                        The number of CPU threads to be used to calculate the initial
             *                                          statistics in parallel. Must be at least 1
             */
            LabelWiseStatisticsProviderFactory(
                std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
                std::shared_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
                std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, uint32 numThreads);

            std::unique_ptr<IStatisticsProvider> create(
                std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr) const override;

    };

}
