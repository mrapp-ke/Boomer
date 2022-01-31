/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/statistics/statistics_provider.hpp"
#include "boosting/statistics/statistics_label_wise.hpp"


namespace boosting {

    /**
     * Provides access to an object of type `ILabelWiseStatistics`.
     *
     * @tparam RuleEvaluationFactory The type of the classes that may be used for calculating the predictions, as well
     *                               as corresponding quality scores, of rules
     */
    template<typename RuleEvaluationFactory>
    class LabelWiseStatisticsProvider final : public IStatisticsProvider {

        private:

            const RuleEvaluationFactory& regularRuleEvaluationFactory_;

            const RuleEvaluationFactory& pruningRuleEvaluationFactory_;

            std::unique_ptr<ILabelWiseStatistics<RuleEvaluationFactory>> statisticsPtr_;

        public:

            /**
             * @param regularRuleEvaluationFactory  A reference to an object of template type `RuleEvaluationFactory` to
             *                                      switch to when invoking the function `switchToRegularRuleEvaluation`
             * @param pruningRuleEvaluationFactory  A reference to an object of template type `RuleEvaluationFactory` to
             *                                      switch to when invoking the function `switchToPruningRuleEvaluation`
             * @param statisticsPtr                 An unique pointer to an object of type `ILabelWiseStatistics` to
             *                                      provide access to
             */
            LabelWiseStatisticsProvider(const RuleEvaluationFactory& regularRuleEvaluationFactory,
                                        const RuleEvaluationFactory& pruningRuleEvaluationFactory,
                                        std::unique_ptr<ILabelWiseStatistics<RuleEvaluationFactory>> statisticsPtr)
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

}
