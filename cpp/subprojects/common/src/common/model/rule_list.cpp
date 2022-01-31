#include "common/model/rule_list.hpp"
#include "common/model/body_empty.hpp"
#include "common/output/predictor_classification.hpp"
#include "common/output/predictor_regression.hpp"
#include "common/output/predictor_probability.hpp"
#include "common/output/label_space_info.hpp"


RuleList::Rule::Rule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr)
    : bodyPtr_(std::move(bodyPtr)), headPtr_(std::move(headPtr)) {

}

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

RuleList::RuleConstIterator::RuleConstIterator(const std::forward_list<Rule>& list, uint32 index)
    : iterator_(list.cbegin()), index_(index) {

}

RuleList::RuleConstIterator::reference RuleList::RuleConstIterator::operator*() const {
    return *iterator_;
}

RuleList::RuleConstIterator& RuleList::RuleConstIterator::operator++() {
    ++iterator_;
    ++index_;
    return *this;
}

RuleList::RuleConstIterator& RuleList::RuleConstIterator::operator++(int n) {
    iterator_++;
    index_++;
    return *this;
}

bool RuleList::RuleConstIterator::operator!=(const RuleConstIterator& rhs) const {
    return index_ != rhs.index_;
}

bool RuleList::RuleConstIterator::operator==(const RuleConstIterator& rhs) const {
    return index_ == rhs.index_;
}

RuleList::RuleList()
    : it_(list_.begin()), numRules_(0), numUsedRules_(0), containsDefaultRule_(false) {

}

RuleList::const_iterator RuleList::cbegin() const {
    return list_.cbegin();
}

RuleList::const_iterator RuleList::cend() const {
    return list_.cend();
}

RuleList::used_const_iterator RuleList::used_cbegin() const {
    return RuleConstIterator(list_, 0);
}

RuleList::used_const_iterator RuleList::used_cend() const {
    return RuleConstIterator(list_, this->getNumUsedRules());
}

uint32 RuleList::getNumRules() const {
    return numRules_;
}

uint32 RuleList::getNumUsedRules() const {
    return numUsedRules_ > 0 ? numUsedRules_ : numRules_;;
}

void RuleList::setNumUsedRules(uint32 numUsedRules) {
    numUsedRules_ = numUsedRules;
}

void RuleList::addDefaultRule(std::unique_ptr<IHead> headPtr) {
    containsDefaultRule_ = true;
    this->addRule(std::make_unique<EmptyBody>(), std::move(headPtr));
}

void RuleList::addRule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr) {
    if (numRules_ > 0) {
        it_ = list_.emplace_after(it_, std::move(bodyPtr), std::move(headPtr));
    } else {
        list_.emplace_front(std::move(bodyPtr), std::move(headPtr));
        it_ = list_.begin();
    }

    numRules_++;
}

bool RuleList::containsDefaultRule() const {
    return containsDefaultRule_;
}

void RuleList::visit(IBody::EmptyBodyVisitor emptyBodyVisitor, IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                      IHead::CompleteHeadVisitor completeHeadVisitor,
                      IHead::PartialHeadVisitor partialHeadVisitor) const {
    for (auto it = list_.cbegin(); it != list_.cend(); it++) {
        const Rule& rule = *it;
        rule.visit(emptyBodyVisitor, conjunctiveBodyVisitor, completeHeadVisitor, partialHeadVisitor);
    }
}

void RuleList::visitUsed(IBody::EmptyBodyVisitor emptyBodyVisitor,
                          IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                          IHead::CompleteHeadVisitor completeHeadVisitor,
                          IHead::PartialHeadVisitor partialHeadVisitor) const {
    for (auto it = this->used_cbegin(); it != this->used_cend(); it++) {
        const Rule& rule = *it;
        rule.visit(emptyBodyVisitor, conjunctiveBodyVisitor, completeHeadVisitor, partialHeadVisitor);
    }
}

std::unique_ptr<IClassificationPredictor> RuleList::createClassificationPredictor(
        const IClassificationPredictorFactory& factory, const ILabelSpaceInfo& labelSpaceInfo) const {
    return labelSpaceInfo.createClassificationPredictor(factory, *this);
}

std::unique_ptr<IRegressionPredictor> RuleList::createRegressionPredictor(
        const IRegressionPredictorFactory& factory, const ILabelSpaceInfo& labelSpaceInfo) const {
    return labelSpaceInfo.createRegressionPredictor(factory, *this);
}

std::unique_ptr<IProbabilityPredictor> RuleList::createProbabilityPredictor(
        const IProbabilityPredictorFactory& factory, const ILabelSpaceInfo& labelSpaceInfo) const {
    return labelSpaceInfo.createProbabilityPredictor(factory, *this);
}

std::unique_ptr<IRuleList> createRuleList() {
    return std::make_unique<RuleList>();
}
