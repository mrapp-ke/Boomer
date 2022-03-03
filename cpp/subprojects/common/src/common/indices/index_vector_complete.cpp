#include "common/indices/index_vector_complete.hpp"
#include "common/statistics/statistics_immutable.hpp"
#include "common/thresholds/thresholds_subset.hpp"


CompleteIndexVector::CompleteIndexVector(uint32 numElements) {
    numElements_ = numElements;
}

bool CompleteIndexVector::isPartial() const {
    return false;
}

uint32 CompleteIndexVector::getNumElements() const {
    return numElements_;
}

void CompleteIndexVector::setNumElements(uint32 numElements, bool freeMemory) {
    numElements_ = numElements;
}

uint32 CompleteIndexVector::getIndex(uint32 pos) const {
    return pos;
}

CompleteIndexVector::const_iterator CompleteIndexVector::cbegin() const {
    return IndexIterator();
}

CompleteIndexVector::const_iterator CompleteIndexVector::cend() const {
    return IndexIterator(numElements_);
}

std::unique_ptr<IStatisticsSubset> CompleteIndexVector::createSubset(const IImmutableStatistics& statistics) const {
    return statistics.createSubset(*this);
}

std::unique_ptr<IRuleRefinement> CompleteIndexVector::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                           uint32 featureIndex) const {
    return thresholdsSubset.createRuleRefinement(*this, featureIndex);
}
