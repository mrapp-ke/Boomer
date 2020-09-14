#include "statistics.h"

using namespace boosting;


AbstractGradientStatistics::AbstractGradientStatistics(uint32 numStatistics, uint32 numLabels)
    : AbstractStatistics(numStatistics, numLabels) {

}

void AbstractGradientStatistics::resetSampledStatistics() {
    // This function is equivalent to the function `resetCoveredStatistics`...
    this->resetCoveredStatistics();
}

void AbstractGradientStatistics::addSampledStatistic(uint32 statisticIndex, uint32 weight) {
    // This function is equivalent to the function `updateCoveredStatistic`...
    this->updateCoveredStatistic(statisticIndex, weight, false);
}
