#include "common/rule_refinement/refinement_comparator_fixed.hpp"

#include "common/rule_refinement/score_processor.hpp"

#include <algorithm>

FixedRefinementComparator::FixedRefinementComparator(RuleCompareFunction ruleCompareFunction, uint32 maxRefinements,
                                                     const Quality& minQuality)
    : ruleCompareFunction_(ruleCompareFunction), maxRefinements_(maxRefinements),
      refinements_(new Refinement[maxRefinements]), minQuality_(minQuality) {
    order_.reserve(maxRefinements);
}

FixedRefinementComparator::FixedRefinementComparator(RuleCompareFunction ruleCompareFunction, uint32 maxRefinements)
    : FixedRefinementComparator(ruleCompareFunction, maxRefinements, ruleCompareFunction.minQuality) {}

FixedRefinementComparator::FixedRefinementComparator(const FixedRefinementComparator& comparator)
    : FixedRefinementComparator(comparator.ruleCompareFunction_, comparator.maxRefinements_, comparator.minQuality_) {}

FixedRefinementComparator::~FixedRefinementComparator() {
    delete[] refinements_;
}

uint32 FixedRefinementComparator::getNumElements() const {
    return (uint32) order_.size();
}

FixedRefinementComparator::iterator FixedRefinementComparator::begin() {
    return order_.begin();
}

FixedRefinementComparator::iterator FixedRefinementComparator::end() {
    return order_.end();
}

bool FixedRefinementComparator::isImprovement(const IScoreVector& scoreVector) const {
    return ruleCompareFunction_.compare(scoreVector, minQuality_);
}

void FixedRefinementComparator::pushRefinement(const Refinement& refinement, const IScoreVector& scoreVector) {
    auto numRefinements = order_.size();

    if (numRefinements < maxRefinements_) {
        Refinement& newRefinement = refinements_[numRefinements];
        newRefinement = refinement;
        ScoreProcessor scoreProcessor(newRefinement.headPtr);
        scoreProcessor.processScores(scoreVector);
        order_.push_back(newRefinement);
    } else {
        Refinement& worstRefinement = order_.back();
        worstRefinement = refinement;
        ScoreProcessor scoreProcessor(worstRefinement.headPtr);
        scoreProcessor.processScores(scoreVector);
    }

    std::sort(order_.begin(), order_.end(), [=](const Refinement& a, const Refinement& b) {
        return ruleCompareFunction_.compare(*a.headPtr, *b.headPtr);
    });

    const Refinement& worstRefinement = order_.back();
    minQuality_ = *worstRefinement.headPtr;
}

bool FixedRefinementComparator::merge(FixedRefinementComparator& comparator) {
    bool result = false;
    Refinement* tmp = new Refinement[maxRefinements_];
    uint32 n = 0;

    auto it1 = order_.begin();
    auto end1 = order_.end();
    auto it2 = comparator.order_.begin();
    auto end2 = comparator.order_.end();

    while (n < maxRefinements_ && it1 != end1 && it2 != end2) {
        Refinement& refinement1 = *it1;
        Refinement& refinement2 = *it2;
        Refinement& newRefinement = tmp[n];

        if (ruleCompareFunction_.compare(*refinement1.headPtr, *refinement2.headPtr)) {
            newRefinement = refinement1;
            newRefinement.headPtr = std::move(refinement1.headPtr);
            it1++;
        } else {
            result = true;
            newRefinement = refinement2;
            newRefinement.headPtr = std::move(refinement2.headPtr);
            it2++;
        }

        n++;
    }

    for (; n < maxRefinements_ && it1 != end1; it1++) {
        Refinement& refinement = *it1;
        Refinement& newRefinement = tmp[n];
        newRefinement = refinement;
        newRefinement.headPtr = std::move(refinement.headPtr);
        n++;
    }

    for (; n < maxRefinements_ && it2 != end2; it2++) {
        result = true;
        Refinement& refinement = *it2;
        Refinement& newRefinement = tmp[n];
        newRefinement = refinement;
        newRefinement.headPtr = std::move(refinement.headPtr);
        n++;
    }

    order_.clear();

    for (uint32 i = 0; i < n; i++) {
        Refinement& newRefinement = tmp[i];
        order_.push_back(newRefinement);
    }

    if (n > 0) {
        const Refinement& worstRefinement = order_.back();
        minQuality_ = *worstRefinement.headPtr;
    }

    delete[] refinements_;
    refinements_ = tmp;
    return result;
}
