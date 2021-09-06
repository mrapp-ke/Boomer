#include "common/head_refinement/prediction_full.hpp"
#include "common/head_refinement/head_refinement.hpp"
#include "common/rule_refinement/rule_refinement.hpp"
#include "common/statistics/statistics.hpp"
#include "common/model/head_full.hpp"


FullPrediction::FullPrediction(uint32 numElements)
    : AbstractEvaluatedPrediction(numElements), indexVector_(FullIndexVector(numElements)) {

}

FullPrediction::index_const_iterator FullPrediction::indices_cbegin() const {
    return indexVector_.cbegin();
}

FullPrediction::index_const_iterator FullPrediction::indices_cend() const {
    return indexVector_.cend();
}

void FullPrediction::setNumElements(uint32 numElements, bool freeMemory) {
    AbstractPrediction::setNumElements(numElements, freeMemory);
    indexVector_.setNumElements(numElements, freeMemory);
}

bool FullPrediction::isPartial() const {
    return false;
}

uint32 FullPrediction::getIndex(uint32 pos) const {
    return indexVector_.getIndex(pos);
}

std::unique_ptr<IStatisticsSubset> FullPrediction::createSubset(const IImmutableStatistics& statistics) const {
    return indexVector_.createSubset(statistics);
}

std::unique_ptr<IRuleRefinement> FullPrediction::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                      uint32 featureIndex) const {
    return indexVector_.createRuleRefinement(thresholdsSubset, featureIndex);
}

std::unique_ptr<IHeadRefinement> FullPrediction::createHeadRefinement(const IHeadRefinementFactory& factory) const {
    return indexVector_.createHeadRefinement(factory);
}

void FullPrediction::apply(IStatistics& statistics, uint32 statisticIndex) const {
    statistics.applyPrediction(statisticIndex, *this);
}

std::unique_ptr<IHead> FullPrediction::toHead() const {
    return std::make_unique<FullHead>(*this);
}
