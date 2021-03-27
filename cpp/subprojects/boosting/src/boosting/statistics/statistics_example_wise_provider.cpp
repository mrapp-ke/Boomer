#include "boosting/statistics/statistics_example_wise_provider.hpp"
#include "boosting/statistics/statistics_example_wise_dense.hpp"


namespace boosting {

    /**
     * Provides access to an object of type `IExampleWiseStatistics`.
     */
    class ExampleWiseStatisticsProvider : public IStatisticsProvider {

        private:

            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            std::unique_ptr<IExampleWiseStatistics> statisticsPtr_;

        public:

            /**
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type
             *                                  `IExampleWiseRuleEvaluationFactory` to switch to when invoking the
             *                                  function `switchRuleEvaluation`
             * @param statisticsPtr             An unique pointer to an object of type `IExampleWiseStatistics` to
             *                                  provide access to
             */
            ExampleWiseStatisticsProvider(std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                          std::unique_ptr<IExampleWiseStatistics> statisticsPtr)
                : ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr), statisticsPtr_(std::move(statisticsPtr)) {

            }

            IStatistics& get() const override {
                return *statisticsPtr_;
            }

            void switchRuleEvaluation() override {
                statisticsPtr_->setRuleEvaluationFactory(ruleEvaluationFactoryPtr_);
            }

    };

    ExampleWiseStatisticsProviderFactory::ExampleWiseStatisticsProviderFactory(
            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
            std::shared_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, uint32 numThreads)
        : lossFunctionPtr_(lossFunctionPtr), defaultRuleEvaluationFactoryPtr_(defaultRuleEvaluationFactoryPtr),
          ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr), numThreads_(numThreads) {

    }

    std::unique_ptr<IStatisticsProvider> ExampleWiseStatisticsProviderFactory::create(
            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr) const {
        DenseExampleWiseStatisticsFactory statisticsFactory(lossFunctionPtr_, defaultRuleEvaluationFactoryPtr_,
                                                            labelMatrixPtr, numThreads_);
        return std::make_unique<ExampleWiseStatisticsProvider>(ruleEvaluationFactoryPtr_, statisticsFactory.create());
    }

}
