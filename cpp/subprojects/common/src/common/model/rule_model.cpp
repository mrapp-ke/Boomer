#include "common/model/rule_model.hpp"

RuleModel::RuleConstIterator::RuleConstIterator(const std::forward_list<Rule>& list, uint32 index)
    : iterator_(list.cbegin()), index_(index) {

}

RuleModel::RuleConstIterator::reference RuleModel::RuleConstIterator::operator*() const {
    return *iterator_;
}

RuleModel::RuleConstIterator& RuleModel::RuleConstIterator::operator++() {
    ++iterator_;
    ++index_;
    return *this;
}

RuleModel::RuleConstIterator& RuleModel::RuleConstIterator::operator++(int n) {
    iterator_++;
    index_++;
    return *this;
}

bool RuleModel::RuleConstIterator::operator!=(const RuleConstIterator& rhs) const {
    return index_ != rhs.index_;
}

bool RuleModel::RuleConstIterator::operator==(const RuleConstIterator& rhs) const {
    return index_ == rhs.index_;
}

RuleModel::RuleModel()
    : it_(list_.begin()), numRules_(0), numUsedRules_(0) {

}

RuleModel::const_iterator RuleModel::cbegin() const {
    return list_.cbegin();
}

RuleModel::const_iterator RuleModel::cend() const {
    return list_.cend();
}

RuleModel::used_const_iterator RuleModel::used_cbegin() const {
    return RuleConstIterator(list_, 0);
}

RuleModel::used_const_iterator RuleModel::used_cend() const {
    return RuleConstIterator(list_, this->getNumUsedRules());
}

uint32 RuleModel::getNumRules() const {
    return numRules_;
}

uint32 RuleModel::getNumUsedRules() const {
    return numUsedRules_ > 0 ? numUsedRules_ : numRules_;;
}

void RuleModel::setNumUsedRules(uint32 numUsedRules) {
    numUsedRules_ = numUsedRules;
}

void RuleModel::addRule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr) {
    if (numRules_ > 0) {
        it_ = list_.emplace_after(it_, std::move(bodyPtr), std::move(headPtr));
    } else {
        list_.emplace_front(std::move(bodyPtr), std::move(headPtr));
        it_ = list_.begin();
    }

    numRules_++;
}

void RuleModel::visit(IBody::EmptyBodyVisitor emptyBodyVisitor, IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                      IHead::CompleteHeadVisitor completeHeadVisitor,
                      IHead::PartialHeadVisitor partialHeadVisitor) const {
    for (auto it = list_.cbegin(); it != list_.cend(); it++) {
        const Rule& rule = *it;
        rule.visit(emptyBodyVisitor, conjunctiveBodyVisitor, completeHeadVisitor, partialHeadVisitor);
    }
}

void RuleModel::visitUsed(IBody::EmptyBodyVisitor emptyBodyVisitor,
                          IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                          IHead::CompleteHeadVisitor completeHeadVisitor,
                          IHead::PartialHeadVisitor partialHeadVisitor) const {
    for (auto it = this->used_cbegin(); it != this->used_cend(); it++) {
        const Rule& rule = *it;
        rule.visit(emptyBodyVisitor, conjunctiveBodyVisitor, completeHeadVisitor, partialHeadVisitor);
    }
}
