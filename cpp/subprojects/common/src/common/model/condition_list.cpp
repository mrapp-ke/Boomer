#include "common/model/condition_list.hpp"


ConditionList::const_iterator ConditionList::cbegin() const {
    return list_.cbegin();
}

ConditionList::const_iterator ConditionList::cend() const {
    return list_.cend();
}

ConditionList::size_type ConditionList::getNumConditions() const {
    return list_.size();
}

uint32 ConditionList::getNumConditions(Comparator comparator) const {
    return numConditionsPerComparator_[comparator];
}

void ConditionList::addCondition(const Condition& condition) {
    numConditionsPerComparator_[condition.comparator] += 1;
    list_.emplace_back(condition);
}

void ConditionList::removeLast() {
    Condition& condition = list_.back();
    numConditionsPerComparator_[condition.comparator] -= 1;
    list_.pop_back();
};
