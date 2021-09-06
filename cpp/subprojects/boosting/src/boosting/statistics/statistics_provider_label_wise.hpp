#include "common/statistics/statistics_provider.hpp"
#include "boosting/statistics/statistics_label_wise.hpp"


namespace boosting {

    /**
     * Provides access to an object of type `ILabelWiseStatistics`.
     */
    class LabelWiseStatisticsProvider : public IStatisticsProvider {

        private:

            const ILabelWiseRuleEvaluationFactory& regularRuleEvaluationFactory_;

            const ILabelWiseRuleEvaluationFactory& pruningRuleEvaluationFactory_;

            std::unique_ptr<ILabelWiseStatistics> statisticsPtr_;

        public:

            /**
             * @param regularRuleEvaluationFactory  A reference to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                      to switch to when invoking the function
             *                                      `switchToRegularRuleEvaluation`
             * @param pruningRuleEvaluationFactory  A reference to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                      to switch to when invoking the function
             *                                      `switchToPruningRuleEvaluation`
             * @param statisticsPtr                 An unique pointer to an object of type `ILabelWiseStatistics` to
             *                                      provide access to
             */
            LabelWiseStatisticsProvider(const ILabelWiseRuleEvaluationFactory& regularRuleEvaluationFactory,
                                        const ILabelWiseRuleEvaluationFactory& pruningRuleEvaluationFactory,
                                        std::unique_ptr<ILabelWiseStatistics> statisticsPtr)
                : regularRuleEvaluationFactory_(regularRuleEvaluationFactory),
                  pruningRuleEvaluationFactory_(pruningRuleEvaluationFactory),
                  statisticsPtr_(std::move(statisticsPtr)) {

            }

            IStatistics& get() const override {
                return *statisticsPtr_;
            }

            void switchToRegularRuleEvaluation() override {
                statisticsPtr_->setRuleEvaluationFactory(regularRuleEvaluationFactory_);
            }

            void switchToPruningRuleEvaluation() override {
                statisticsPtr_->setRuleEvaluationFactory(pruningRuleEvaluationFactory_);
            }

    };

}
