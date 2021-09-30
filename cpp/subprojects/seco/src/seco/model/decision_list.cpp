#include "seco/model/decision_list.hpp"
#include "common/model/body_empty.hpp"
#include "common/model/body_conjunctive.hpp"


namespace seco {

    DecisionListBuilder::DecisionListBuilder()
        : modelPtr_(std::make_unique<RuleModel>()){

    }

    void DecisionListBuilder::setDefaultRule(const AbstractPrediction& prediction) {
        defaultHeadPtr_ = prediction.toHead();
    }

    void DecisionListBuilder::addRule(const ConditionList& conditions, const AbstractPrediction& prediction) {
        modelPtr_->addRule(std::make_unique<ConjunctiveBody>(conditions), prediction.toHead());
    }

    std::unique_ptr<RuleModel> DecisionListBuilder::build(uint32 numUsedRules) {
        if (defaultHeadPtr_.get() != nullptr) {
            modelPtr_->addRule(std::make_unique<EmptyBody>(), std::move(defaultHeadPtr_));
        }

        modelPtr_->setNumUsedRules(numUsedRules);
        return std::move(modelPtr_);
    }

}

