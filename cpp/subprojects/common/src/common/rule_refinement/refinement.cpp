#include "common/rule_refinement/refinement.hpp"


bool Refinement::isBetterThan(const Refinement& another) const {
    const AbstractEvaluatedPrediction* head = headPtr.get();

    if (head != nullptr) {
        const AbstractEvaluatedPrediction* anotherHead = another.headPtr.get();
        return anotherHead == nullptr || head->overallQualityScore < anotherHead->overallQualityScore;
    }

    return false;
}
