#include "common/model/condition_list.hpp"

ConditionList::ConditionList() : numConditionsPerComparator_({0, 0, 0, 0}) {}

ConditionList::ConditionList(const ConditionList& conditionList)
    : vector_(conditionList.vector_),
      numConditionsPerComparator_(
        {conditionList.numConditionsPerComparator_[0], conditionList.numConditionsPerComparator_[1],
         conditionList.numConditionsPerComparator_[2], conditionList.numConditionsPerComparator_[3]}) {}

ConditionList::const_iterator ConditionList::cbegin() const {
    return vector_.cbegin();
}

ConditionList::const_iterator ConditionList::cend() const {
    return vector_.cend();
}

uint32 ConditionList::getNumConditions() const {
    return (uint32) vector_.size();
}

void ConditionList::addCondition(const Condition& condition) {
    numConditionsPerComparator_[condition.comparator] += 1;
    vector_.emplace_back(condition);
}

void ConditionList::removeLastCondition() {
    const Condition& condition = vector_.back();
    numConditionsPerComparator_[condition.comparator] -= 1;
    vector_.pop_back();
};

std::unique_ptr<ConjunctiveBody> ConditionList::createConjunctiveBody() const {
    std::unique_ptr<ConjunctiveBody> bodyPtr =
      std::make_unique<ConjunctiveBody>(numConditionsPerComparator_[LEQ], numConditionsPerComparator_[GR],
                                        numConditionsPerComparator_[EQ], numConditionsPerComparator_[NEQ]);
    uint32 leqIndex = 0;
    uint32 grIndex = 0;
    uint32 eqIndex = 0;
    uint32 neqIndex = 0;

    for (auto it = vector_.cbegin(); it != vector_.cend(); it++) {
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
            default: {
                break;
            }
        }
    }

    return bodyPtr;
}
