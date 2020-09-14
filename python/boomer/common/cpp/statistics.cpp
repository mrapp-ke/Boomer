#include "statistics.h"


AbstractRefinementSearch::~AbstractRefinementSearch() {

}

void AbstractRefinementSearch::updateSearch(uint32 statisticIndex, uint32 weight) {

}

void AbstractRefinementSearch::resetSearch() {

}

LabelWisePredictionCandidate* AbstractRefinementSearch::calculateLabelWisePrediction(bool uncovered, bool accumulated) {
    return NULL;
}

PredictionCandidate* AbstractRefinementSearch::calculateExampleWisePrediction(bool uncovered, bool accumulated) {
    return NULL;
}

PredictionCandidate* AbstractDecomposableRefinementSearch::calculateExampleWisePrediction(bool uncovered,
                                                                                          bool accumulated) {
    // In the decomposable case, the example-wise predictions are the same as the label-wise predictions...
    return (PredictionCandidate*) this->calculateLabelWisePrediction(uncovered, accumulated);
}

AbstractStatistics::AbstractStatistics(uint32 numStatistics, uint32 numLabels) {
    numStatistics_ = numStatistics;
    numLabels_ = numLabels;
}

AbstractStatistics::~AbstractStatistics() {

}

void AbstractStatistics::resetSampledStatistics() {

}

void AbstractStatistics::addSampledStatistic(uint32 statisticIndex, uint32 weight) {

}

void AbstractStatistics::resetCoveredStatistics() {

}

void AbstractStatistics::updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) {

}

AbstractRefinementSearch* AbstractStatistics::beginSearch(uint32 numLabelIndices, const uint32* labelIndices) {
    return NULL;
}

void AbstractStatistics::applyPrediction(uint32 statisticIndex, Prediction* prediction) {

}
