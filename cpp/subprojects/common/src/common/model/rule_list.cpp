#include "common/model/rule_list.hpp"

#include "common/model/body_empty.hpp"
#include "common/prediction/label_space_info.hpp"
#include "common/prediction/predictor_binary.hpp"
#include "common/prediction/predictor_probability.hpp"
#include "common/prediction/predictor_score.hpp"

RuleList::Rule::Rule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr)
    : bodyPtr_(std::move(bodyPtr)), headPtr_(std::move(headPtr)) {}

const IBody& RuleList::Rule::getBody() const {
    return *bodyPtr_;
}

const IHead& RuleList::Rule::getHead() const {
    return *headPtr_;
}

void RuleList::Rule::visit(IBody::EmptyBodyVisitor emptyBodyVisitor,
                           IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                           IHead::CompleteHeadVisitor completeHeadVisitor,
                           IHead::PartialHeadVisitor partialHeadVisitor) const {
    bodyPtr_->visit(emptyBodyVisitor, conjunctiveBodyVisitor);
    headPtr_->visit(completeHeadVisitor, partialHeadVisitor);
}

RuleList::ConstIterator::ConstIterator(bool defaultRuleTakesPrecedence, const Rule* defaultRule,
                                       std::vector<Rule>::const_iterator iterator, uint32 start, uint32 end)
    : defaultRule_(defaultRule), iterator_(iterator),
      offset_(defaultRuleTakesPrecedence && defaultRule != nullptr ? 1 : 0),
      defaultRuleIndex_(offset_ > 0 ? 0 : end - (defaultRule != nullptr ? 1 : 0)), index_(start) {}

RuleList::ConstIterator::reference RuleList::ConstIterator::operator*() const {
    uint32 index = index_;

    if (index == defaultRuleIndex_) {
        return *defaultRule_;
    } else {
        return iterator_[index - offset_];
    }
}

RuleList::ConstIterator& RuleList::ConstIterator::operator++() {
    ++index_;
    return *this;
}

RuleList::ConstIterator& RuleList::ConstIterator::operator++(int n) {
    index_++;
    return *this;
}

RuleList::ConstIterator RuleList::ConstIterator::operator+(const uint32 difference) const {
    ConstIterator iterator(*this);
    iterator += difference;
    return iterator;
}

RuleList::ConstIterator& RuleList::ConstIterator::operator+=(const uint32 difference) {
    index_ += difference;
    return *this;
}

bool RuleList::ConstIterator::operator!=(const ConstIterator& rhs) const {
    return index_ != rhs.index_;
}

bool RuleList::ConstIterator::operator==(const ConstIterator& rhs) const {
    return index_ == rhs.index_;
}

RuleList::ConstIterator::difference_type RuleList::ConstIterator::operator-(const ConstIterator& rhs) const {
    return index_ - rhs.index_;
}

RuleList::RuleList(bool defaultRuleTakesPrecedence)
    : numUsedRules_(0), defaultRuleTakesPrecedence_(defaultRuleTakesPrecedence) {}

RuleList::const_iterator RuleList::cbegin(uint32 maxRules) const {
    uint32 numRules = maxRules > 0 ? std::min(this->getNumRules(), maxRules) : this->getNumRules();
    return ConstIterator(defaultRuleTakesPrecedence_, defaultRulePtr_.get(), ruleList_.cbegin(), 0, numRules);
}

RuleList::const_iterator RuleList::cend(uint32 maxRules) const {
    uint32 numRules = maxRules > 0 ? std::min(this->getNumRules(), maxRules) : this->getNumRules();
    return ConstIterator(defaultRuleTakesPrecedence_, defaultRulePtr_.get(), ruleList_.cbegin(), numRules, numRules);
}

RuleList::const_iterator RuleList::used_cbegin(uint32 maxRules) const {
    uint32 numRules = maxRules > 0 ? std::min(this->getNumUsedRules(), maxRules) : this->getNumUsedRules();
    return ConstIterator(defaultRuleTakesPrecedence_, defaultRulePtr_.get(), ruleList_.cbegin(), 0, numRules);
}

RuleList::const_iterator RuleList::used_cend(uint32 maxRules) const {
    uint32 numRules = maxRules > 0 ? std::min(this->getNumUsedRules(), maxRules) : this->getNumUsedRules();
    return ConstIterator(defaultRuleTakesPrecedence_, defaultRulePtr_.get(), ruleList_.cbegin(), numRules, numRules);
}

uint32 RuleList::getNumRules() const {
    uint32 numRules = (uint32) ruleList_.size();

    if (this->containsDefaultRule()) {
        numRules++;
    }

    return numRules;
}

uint32 RuleList::getNumUsedRules() const {
    return numUsedRules_ > 0 ? numUsedRules_ : this->getNumRules();
}

void RuleList::setNumUsedRules(uint32 numUsedRules) {
    numUsedRules_ = numUsedRules;
}

void RuleList::addDefaultRule(std::unique_ptr<IHead> headPtr) {
    defaultRulePtr_ = std::make_unique<Rule>(std::make_unique<EmptyBody>(), std::move(headPtr));
}

void RuleList::addRule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr) {
    ruleList_.emplace_back(std::move(bodyPtr), std::move(headPtr));
}

bool RuleList::containsDefaultRule() const {
    return defaultRulePtr_ != nullptr;
}

bool RuleList::isDefaultRuleTakingPrecedence() const {
    return defaultRuleTakesPrecedence_;
}

void RuleList::visit(IBody::EmptyBodyVisitor emptyBodyVisitor, IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                     IHead::CompleteHeadVisitor completeHeadVisitor,
                     IHead::PartialHeadVisitor partialHeadVisitor) const {
    for (auto it = this->cbegin(); it != this->cend(); it++) {
        const Rule& rule = *it;
        rule.visit(emptyBodyVisitor, conjunctiveBodyVisitor, completeHeadVisitor, partialHeadVisitor);
    }
}

void RuleList::visitUsed(IBody::EmptyBodyVisitor emptyBodyVisitor, IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                         IHead::CompleteHeadVisitor completeHeadVisitor,
                         IHead::PartialHeadVisitor partialHeadVisitor) const {
    for (auto it = this->used_cbegin(); it != this->used_cend(); it++) {
        const Rule& rule = *it;
        rule.visit(emptyBodyVisitor, conjunctiveBodyVisitor, completeHeadVisitor, partialHeadVisitor);
    }
}

std::unique_ptr<IBinaryPredictor> RuleList::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
  const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return labelSpaceInfo.createBinaryPredictor(factory, featureMatrix, *this, marginalProbabilityCalibrationModel,
                                                jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IBinaryPredictor> RuleList::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return labelSpaceInfo.createBinaryPredictor(factory, featureMatrix, *this, marginalProbabilityCalibrationModel,
                                                jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ISparseBinaryPredictor> RuleList::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
  const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return labelSpaceInfo.createSparseBinaryPredictor(
      factory, featureMatrix, *this, marginalProbabilityCalibrationModel, jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ISparseBinaryPredictor> RuleList::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix,
  const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return labelSpaceInfo.createSparseBinaryPredictor(
      factory, featureMatrix, *this, marginalProbabilityCalibrationModel, jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IScorePredictor> RuleList::createScorePredictor(const IScorePredictorFactory& factory,
                                                                const CContiguousFeatureMatrix& featureMatrix,
                                                                const ILabelSpaceInfo& labelSpaceInfo,
                                                                uint32 numLabels) const {
    return labelSpaceInfo.createScorePredictor(factory, featureMatrix, *this, numLabels);
}

std::unique_ptr<IScorePredictor> RuleList::createScorePredictor(const IScorePredictorFactory& factory,
                                                                const CsrFeatureMatrix& featureMatrix,
                                                                const ILabelSpaceInfo& labelSpaceInfo,
                                                                uint32 numLabels) const {
    return labelSpaceInfo.createScorePredictor(factory, featureMatrix, *this, numLabels);
}

std::unique_ptr<IProbabilityPredictor> RuleList::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
  const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return labelSpaceInfo.createProbabilityPredictor(factory, featureMatrix, *this, marginalProbabilityCalibrationModel,
                                                     jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IProbabilityPredictor> RuleList::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix,
  const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return labelSpaceInfo.createProbabilityPredictor(factory, featureMatrix, *this, marginalProbabilityCalibrationModel,
                                                     jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IRuleList> createRuleList(bool defaultRuleTakesPrecedence) {
    return std::make_unique<RuleList>(defaultRuleTakesPrecedence);
}
