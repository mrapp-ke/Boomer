/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/statistics/statistics_provider.hpp"
#include "boosting/statistics/statistics_example_wise.hpp"
#include "boosting/statistics/statistics_label_wise.hpp"


namespace boosting {

    /**
     * Provides access to an object of type `IExampleWiseStatistics`.
     *
     * @tparam LabelWiseRuleEvaluationFactory   The type of the classes that may be used for calculating the label-wise
     *                                          predictions, as well as corresponding quality scores, of rules
     * @tparam ExampleWiseRuleEvaluationFactory The type of the classes that may be used for calculating the
     *                                          example-wise predictions, as well as corresponding quality scores, of
     *                                          rules
     */
    template<typename ExampleWiseRuleEvaluationFactory, typename LabelWiseRuleEvaluationFactory>
    class ExampleWiseStatisticsProvider : public IStatisticsProvider {

        private:

            const ExampleWiseRuleEvaluationFactory& regularRuleEvaluationFactory_;

            const ExampleWiseRuleEvaluationFactory& pruningRuleEvaluationFactory_;

            std::unique_ptr<IExampleWiseStatistics<ExampleWiseRuleEvaluationFactory, LabelWiseRuleEvaluationFactory>> statisticsPtr_;

        public:

            /**
             * @param regularRuleEvaluationFactory  A reference to an object of template type
             *                                      `ExampleWiseRuleEvaluationFactory` to switch to when invoking the
             *                                      function `switchToRegularRuleEvaluation`
             * @param pruningRuleEvaluationFactory  A reference to an object of template type
             *                                      `ExampleWiseRuleEvaluationFactory` to switch to when invoking the
             *                                      function `switchToPruningRuleEvaluation`
             * @param statisticsPtr                 An unique pointer to an object of type `IExampleWiseStatistics` to
             *                                      provide access to
             */
            ExampleWiseStatisticsProvider(
                    const ExampleWiseRuleEvaluationFactory& regularRuleEvaluationFactory,
                    const ExampleWiseRuleEvaluationFactory& pruningRuleEvaluationFactory,
                    std::unique_ptr<IExampleWiseStatistics<ExampleWiseRuleEvaluationFactory, LabelWiseRuleEvaluationFactory>> statisticsPtr)
                : regularRuleEvaluationFactory_(regularRuleEvaluationFactory),
                  pruningRuleEvaluationFactory_(pruningRuleEvaluationFactory),
                  statisticsPtr_(std::move(statisticsPtr)) {

            }

            /**
             * @see `IStatisticsProvider::get`
             */
            IStatistics& get() const override {
                return *statisticsPtr_;
            }

            /**
             * @see `IStatisticsProvider::switchToRegularRuleEvaluation`
             */
            void switchToRegularRuleEvaluation() override {
                statisticsPtr_->setRuleEvaluationFactory(regularRuleEvaluationFactory_);
            }

            /**
             * @see `IStatisticsProvider::switchToPruningRuleEvaluation`
             */
            void switchToPruningRuleEvaluation() override {
                statisticsPtr_->setRuleEvaluationFactory(pruningRuleEvaluationFactory_);
            }

    };

    /**
     * Provides access to an object of type `IExampleWiseStatistics` that can be converted into an object of type
     * `ILabelWiseStatistics`.
     *
     * @tparam LabelWiseRuleEvaluationFactory   The type of the classes that may be used for calculating the label-wise
     *                                          predictions, as well as corresponding quality scores, of rules
     * @tparam ExampleWiseRuleEvaluationFactory The type of the classes that may be used for calculating the
     *                                          example-wise predictions, as well as corresponding quality scores, of
     *                                          rules
     */
    template<typename ExampleWiseRuleEvaluationFactory, typename LabelWiseRuleEvaluationFactory>
    class ConvertibleExampleWiseStatisticsProvider : public IStatisticsProvider {

        private:

            const LabelWiseRuleEvaluationFactory& regularRuleEvaluationFactory_;

            const LabelWiseRuleEvaluationFactory& pruningRuleEvaluationFactory_;

            std::unique_ptr<IExampleWiseStatistics<ExampleWiseRuleEvaluationFactory, LabelWiseRuleEvaluationFactory>> exampleWiseStatisticsPtr_;

            std::unique_ptr<ILabelWiseStatistics<LabelWiseRuleEvaluationFactory>> labelWiseStatisticsPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param regularRuleEvaluationFactory  A reference to an object of template type
             *                                      `LabelWiseRuleEvaluationFactory` to switch to when invoking the
             *                                      function `switchToRegularRuleEvaluation`
             * @param pruningRuleEvaluationFactory  A reference to an object of template type
             *                                      `LabelWiseRuleEvaluationFactory` to switch to when invoking the
             *                                      function `switchToPruningRuleEvaluation`
             * @param statisticsPtr                 An unique pointer to an object of type `IExampleWiseStatistics` to
             *                                      provide access to
             * @param numThreads                    The number of threads that should be used to convert the statistics
             *                                      for individual examples in parallel
             */
            ConvertibleExampleWiseStatisticsProvider(
                    const LabelWiseRuleEvaluationFactory& regularRuleEvaluationFactory,
                    const LabelWiseRuleEvaluationFactory& pruningRuleEvaluationFactory,
                    std::unique_ptr<IExampleWiseStatistics<ExampleWiseRuleEvaluationFactory, LabelWiseRuleEvaluationFactory>> statisticsPtr,
                    uint32 numThreads)
                : regularRuleEvaluationFactory_(regularRuleEvaluationFactory),
                  pruningRuleEvaluationFactory_(pruningRuleEvaluationFactory),
                  exampleWiseStatisticsPtr_(std::move(statisticsPtr)), numThreads_(numThreads) {

            }

            /**
             * @see `IStatisticsProvider::get`
             */
            IStatistics& get() const override {
                IExampleWiseStatistics<ExampleWiseRuleEvaluationFactory, LabelWiseRuleEvaluationFactory>* exampleWiseStatistics =
                    exampleWiseStatisticsPtr_.get();

                if (exampleWiseStatistics != nullptr) {
                    return *exampleWiseStatistics;
                } else {
                    return *labelWiseStatisticsPtr_;
                }
            }

            /**
             * @see `IStatisticsProvider::switchToRegularRuleEvaluation`
             */
            void switchToRegularRuleEvaluation() override {
                IExampleWiseStatistics<ExampleWiseRuleEvaluationFactory, LabelWiseRuleEvaluationFactory>* exampleWiseStatistics =
                    exampleWiseStatisticsPtr_.get();

                if (exampleWiseStatistics != nullptr) {
                    labelWiseStatisticsPtr_ = exampleWiseStatistics->toLabelWiseStatistics(
                        regularRuleEvaluationFactory_, numThreads_);
                    exampleWiseStatisticsPtr_.reset();
                } else {
                    labelWiseStatisticsPtr_->setRuleEvaluationFactory(regularRuleEvaluationFactory_);
                }
            }

            /**
             * @see `IStatisticsProvider::switchToPruningRuleEvaluation`
             */
            void switchToPruningRuleEvaluation() override {
                IExampleWiseStatistics<ExampleWiseRuleEvaluationFactory, LabelWiseRuleEvaluationFactory>* exampleWiseStatistics =
                    exampleWiseStatisticsPtr_.get();

                if (exampleWiseStatistics != nullptr) {
                    labelWiseStatisticsPtr_ = exampleWiseStatistics->toLabelWiseStatistics(
                        pruningRuleEvaluationFactory_, numThreads_);
                    exampleWiseStatisticsPtr_.reset();
                } else {
                    labelWiseStatisticsPtr_->setRuleEvaluationFactory(pruningRuleEvaluationFactory_);
                }
            }

    };

}
