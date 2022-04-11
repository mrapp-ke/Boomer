#include "boosting/model/rule_list_builder.hpp"


namespace boosting {

    RuleListBuilder::RuleListBuilder()
        : modelPtr_(std::make_unique<RuleList>()) {

    }

    void RuleListBuilder::setDefaultRule(const AbstractPrediction& prediction) {
        modelPtr_->addDefaultRule(prediction.createHead());
    }

    void RuleListBuilder::addRule(const ConditionList& conditions, const AbstractPrediction& prediction) {
        modelPtr_->addRule(conditions.createConjunctiveBody(), prediction.createHead());
    }

    std::unique_ptr<IRuleModel> RuleListBuilder::build(uint32 numUsedRules) {
        modelPtr_->setNumUsedRules(numUsedRules);
        return std::move(modelPtr_);
    }

}
