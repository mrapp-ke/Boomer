#include "boosting/model/rule_list.hpp"
#include "common/model/body_empty.hpp"
#include "common/model/body_conjunctive.hpp"


namespace boosting {

    RuleListBuilder::RuleListBuilder()
        : modelPtr_(std::make_unique<RuleModel>()) {

    }

    void RuleListBuilder::setDefaultRule(const AbstractPrediction& prediction) {
        modelPtr_->addRule(std::make_unique<EmptyBody>(), prediction.toHead());
    }

    void RuleListBuilder::addRule(const ConditionList& conditions, const AbstractPrediction& prediction) {
        modelPtr_->addRule(std::make_unique<ConjunctiveBody>(conditions), prediction.toHead());
    }

    std::unique_ptr<RuleModel> RuleListBuilder::build(uint32 numUsedRules) {
        modelPtr_->setNumUsedRules(numUsedRules);
        return std::move(modelPtr_);
    }

}
