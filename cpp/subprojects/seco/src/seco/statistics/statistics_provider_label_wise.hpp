/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#include "common/statistics/statistics_provider_factory.hpp"
#include "seco/statistics/statistics_label_wise.hpp"


namespace seco {

    /**
     * Provides access to an object of type `ILabelWiseStatistics`.
     *
     * @tparam RuleEvaluationFactory The type of the classes that may be used for calculating the predictions, as well
     *                               as corresponding quality scores, of rules
     */
    template<typename RuleEvaluationFactory>
    class LabelWiseStatisticsProvider : public IStatisticsProvider {

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
