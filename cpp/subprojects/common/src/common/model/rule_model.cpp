#include "common/model/rule_model.hpp"

RuleModel::UsedIterator::UsedIterator(const std::list<Rule>& list, uint32 index)
    : iterator_(list.cbegin()), index_(index) {

}

RuleModel::UsedIterator::reference RuleModel::UsedIterator::operator*() const {
    return *iterator_;
}

RuleModel::UsedIterator& RuleModel::UsedIterator::operator++() {
    ++iterator_;
    ++index_;
    return *this;
}

RuleModel::UsedIterator& RuleModel::UsedIterator::operator++(int n) {
    iterator_++;
    index_++;
    return *this;
}

bool RuleModel::UsedIterator::operator!=(const UsedIterator& rhs) const {
    return index_ != rhs.index_;
}

RuleModel::UsedIterator::difference_type RuleModel::UsedIterator::operator-(const UsedIterator& rhs) const {
    return (difference_type) index_ - (difference_type) rhs.index_;
}

RuleModel::RuleModel()
    : numUsedRules_(0) {

}

RuleModel::const_iterator RuleModel::cbegin() const {
    return list_.cbegin();
}

RuleModel::const_iterator RuleModel::cend() const {
    return list_.cend();
}

RuleModel::used_const_iterator RuleModel::used_cbegin() const {
    return UsedIterator(list_, 0);
}

RuleModel::used_const_iterator RuleModel::used_cend() const {
    return UsedIterator(list_, this->getNumUsedRules());
}

uint32 RuleModel::getNumRules() const {
    return (uint32) list_.size();
}

uint32 RuleModel::getNumUsedRules() const {
    return numUsedRules_ > 0 ? numUsedRules_ : this->getNumRules();
}

void RuleModel::setNumUsedRules(uint32 numUsedRules) {
    numUsedRules_ = numUsedRules;
}

void RuleModel::addRule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr) {
    list_.emplace_back(std::move(bodyPtr), std::move(headPtr));
}

void RuleModel::visit(IBody::EmptyBodyVisitor emptyBodyVisitor, IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                      IHead::FullHeadVisitor fullHeadVisitor, IHead::PartialHeadVisitor partialHeadVisitor) const {
    for (auto it = list_.cbegin(); it != list_.cend(); it++) {
        const Rule& rule = *it;
        rule.visit(emptyBodyVisitor, conjunctiveBodyVisitor, fullHeadVisitor, partialHeadVisitor);
    }
}

void RuleModel::visitUsed(IBody::EmptyBodyVisitor emptyBodyVisitor,
                          IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor, IHead::FullHeadVisitor fullHeadVisitor,
                          IHead::PartialHeadVisitor partialHeadVisitor) const {
    for (auto it = this->used_cbegin(); it != this->used_cend(); it++) {
        const Rule& rule = *it;
        rule.visit(emptyBodyVisitor, conjunctiveBodyVisitor, fullHeadVisitor, partialHeadVisitor);
    }
}
