#include "common/rule_refinement/prediction_partial.hpp"
#include "common/rule_refinement/rule_refinement.hpp"
#include "common/statistics/statistics.hpp"
#include "common/model/head_partial.hpp"


PartialPrediction::PartialPrediction(uint32 numElements)
    : AbstractEvaluatedPrediction(numElements), indexVector_(PartialIndexVector(numElements)) {

}

PartialPrediction::index_iterator PartialPrediction::indices_begin() {
    return indexVector_.begin();
}

PartialPrediction::index_iterator PartialPrediction::indices_end() {
    return indexVector_.end();
}

PartialPrediction::index_const_iterator PartialPrediction::indices_cbegin() const {
    return indexVector_.cbegin();
}

PartialPrediction::index_const_iterator PartialPrediction::indices_cend() const {
    return indexVector_.cend();
}

void PartialPrediction::setNumElements(uint32 numElements, bool freeMemory) {
    AbstractPrediction::setNumElements(numElements, freeMemory);
    indexVector_.setNumElements(numElements, freeMemory);
}

bool PartialPrediction::isPartial() const {
    return true;
}

uint32 PartialPrediction::getIndex(uint32 pos) const {
    return indexVector_.getIndex(pos);
}

std::unique_ptr<IStatisticsSubset> PartialPrediction::createSubset(const IImmutableStatistics& statistics) const {
    return indexVector_.createSubset(statistics);
}

std::unique_ptr<IRuleRefinement> PartialPrediction::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                         uint32 featureIndex) const {
    return indexVector_.createRuleRefinement(thresholdsSubset, featureIndex);
}

void PartialPrediction::apply(IStatistics& statistics, uint32 statisticIndex) const {
    statistics.applyPrediction(statisticIndex, *this);
}

std::unique_ptr<IHead> PartialPrediction::toHead() const {
    return std::make_unique<PartialHead>(*this);
}
