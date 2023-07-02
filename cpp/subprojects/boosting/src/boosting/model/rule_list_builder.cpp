#include "boosting/model/rule_list_builder.hpp"

#include "common/model/rule_list.hpp"

namespace boosting {

    /**
     * Allows to build models that store several rules in the order they have been added.
     */
    class RuleListBuilder final : public IModelBuilder {
        private:

            std::unique_ptr<RuleList> modelPtr_;

        public:

            RuleListBuilder() : modelPtr_(std::make_unique<RuleList>(true)) {}

            /**
             * @see `IModelBuilder::setDefaultRule`
             */
            void setDefaultRule(std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) override {
                modelPtr_->addDefaultRule(predictionPtr->createHead());
            }

            /**
             * @see `IModelBuilder::addRule`
             */
            void addRule(std::unique_ptr<ConditionList>& conditionListPtr,
                         std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) override {
                modelPtr_->addRule(conditionListPtr->createConjunctiveBody(), predictionPtr->createHead());
            }

            /**
             * @see `IModelBuilder::setNumUsedRules`
             */
            void setNumUsedRules(uint32 numUsedRules) override {
                modelPtr_->setNumUsedRules(numUsedRules);
            }

            /**
             * @see `IModelBuilder::buildModel`
             */
            std::unique_ptr<IRuleModel> buildModel() override {
                return std::move(modelPtr_);
            }
    };

    std::unique_ptr<IModelBuilder> RuleListBuilderFactory::create() const {
        return std::make_unique<RuleListBuilder>();
    }

}
