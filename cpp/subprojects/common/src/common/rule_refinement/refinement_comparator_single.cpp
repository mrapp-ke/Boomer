#include "common/rule_refinement/refinement_comparator_single.hpp"

SingleRefinementComparator::SingleRefinementComparator(RuleCompareFunction ruleCompareFunction)
    : ruleCompareFunction_(ruleCompareFunction), bestQuality_(ruleCompareFunction.minQuality),
      scoreProcessor_(ScoreProcessor(bestRefinement_.headPtr)) {}

SingleRefinementComparator::SingleRefinementComparator(const SingleRefinementComparator& comparator)
    : ruleCompareFunction_(comparator.ruleCompareFunction_), bestQuality_(comparator.bestQuality_),
      scoreProcessor_(ScoreProcessor(bestRefinement_.headPtr)) {}

SingleRefinementComparator::iterator SingleRefinementComparator::begin() {
    return &bestRefinement_;
}

SingleRefinementComparator::iterator SingleRefinementComparator::end() {
    return bestRefinement_.headPtr != nullptr ? &bestRefinement_ + 1 : &bestRefinement_;
}

uint32 SingleRefinementComparator::getNumElements() const {
    return bestRefinement_.headPtr != nullptr ? 1 : 0;
}

bool SingleRefinementComparator::isImprovement(const IScoreVector& scoreVector) const {
    return ruleCompareFunction_.compare(scoreVector, bestQuality_);
}

void SingleRefinementComparator::pushRefinement(const Refinement& refinement, const IScoreVector& scoreVector) {
    bestRefinement_ = refinement;
    scoreProcessor_.processScores(scoreVector);
    bestQuality_ = *bestRefinement_.headPtr;
}

bool SingleRefinementComparator::merge(SingleRefinementComparator& comparator) {
    if (ruleCompareFunction_.compare(comparator.bestQuality_, bestQuality_)) {
        Refinement& otherRefinement = comparator.bestRefinement_;
        bestRefinement_ = otherRefinement;
        bestRefinement_.headPtr = std::move(otherRefinement.headPtr);
        bestQuality_ = *bestRefinement_.headPtr;
        return true;
    }

    return false;
}
