#include "common/post_optimization/model_builder_intermediate.hpp"

IntermediateModelBuilder::IntermediateModelBuilder(std::unique_ptr<IModelBuilder> modelBuilderPtr)
    : modelBuilderPtr_(std::move(modelBuilderPtr)), numUsedRules_(0) {}

IntermediateModelBuilder::iterator IntermediateModelBuilder::begin() {
    return intermediateRuleList_.begin();
}

IntermediateModelBuilder::iterator IntermediateModelBuilder::end() {
    return intermediateRuleList_.end();
}

void IntermediateModelBuilder::setDefaultRule(std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) {
    defaultPredictionPtr_ = std::move(predictionPtr);
}

void IntermediateModelBuilder::addRule(std::unique_ptr<ConditionList>& conditionListPtr,
                                       std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) {
    intermediateRuleList_.emplace_back(std::move(conditionListPtr), std::move(predictionPtr));
}

void IntermediateModelBuilder::removeLastRule() {
    intermediateRuleList_.pop_back();
}

uint32 IntermediateModelBuilder::getNumRules() const {
    uint32 numRules = (uint32) intermediateRuleList_.size();

    if (defaultPredictionPtr_) {
        numRules++;
    }

    return numRules;
}

uint32 IntermediateModelBuilder::getNumUsedRules() const {
    return numUsedRules_;
}

void IntermediateModelBuilder::setNumUsedRules(uint32 numUsedRules) {
    numUsedRules_ = numUsedRules;
}

std::unique_ptr<IRuleModel> IntermediateModelBuilder::buildModel() {
    if (defaultPredictionPtr_) {
        modelBuilderPtr_->setDefaultRule(defaultPredictionPtr_);
        defaultPredictionPtr_.release();
    }

    for (auto it = intermediateRuleList_.begin(); it != intermediateRuleList_.end(); it++) {
        IntermediateRule& intermediateRule = *it;
        modelBuilderPtr_->addRule(intermediateRule.first, intermediateRule.second);
    }

    intermediateRuleList_.clear();
    modelBuilderPtr_->setNumUsedRules(numUsedRules_);
    return modelBuilderPtr_->buildModel();
}
