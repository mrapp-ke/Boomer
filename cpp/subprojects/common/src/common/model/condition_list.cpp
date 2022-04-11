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

void ConditionList::addCondition(const Condition& condition) {
    numConditionsPerComparator_[condition.comparator] += 1;
    list_.emplace_back(condition);
}

void ConditionList::removeLast() {
    Condition& condition = list_.back();
    numConditionsPerComparator_[condition.comparator] -= 1;
    list_.pop_back();
};

std::unique_ptr<ConjunctiveBody> ConditionList::createConjunctiveBody() const {
    std::unique_ptr<ConjunctiveBody> bodyPtr =
        std::make_unique<ConjunctiveBody>(numConditionsPerComparator_[LEQ], numConditionsPerComparator_[GR],
                                          numConditionsPerComparator_[EQ], numConditionsPerComparator_[NEQ]);
    uint32 leqIndex = 0;
    uint32 grIndex = 0;
    uint32 eqIndex = 0;
    uint32 neqIndex = 0;

    for (auto it = list_.cbegin(); it != list_.cend(); it++) {
        const Condition& condition = *it;
        uint32 featureIndex = condition.featureIndex;
        float32 threshold = condition.threshold;

        switch (condition.comparator) {
            case LEQ: {
                bodyPtr->leq_indices_begin()[leqIndex] = featureIndex;
                bodyPtr->leq_thresholds_begin()[leqIndex] = threshold;
                leqIndex++;
                break;
            }
            case GR: {
                bodyPtr->gr_indices_begin()[grIndex] = featureIndex;
                bodyPtr->gr_thresholds_begin()[grIndex] = threshold;
                grIndex++;
                break;
            }
            case EQ: {
                bodyPtr->eq_indices_begin()[eqIndex] = featureIndex;
                bodyPtr->eq_thresholds_begin()[eqIndex] = threshold;
                eqIndex++;
                break;
            }
            case NEQ: {
                bodyPtr->neq_indices_begin()[neqIndex] = featureIndex;
                bodyPtr->neq_thresholds_begin()[neqIndex] = threshold;
                neqIndex++;
                break;
            }
            default: { }
        }
    }

    return bodyPtr;
}
