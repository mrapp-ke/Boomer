#include "common/indices/index_vector_partial.hpp"
#include "common/statistics/statistics_immutable.hpp"
#include "common/thresholds/thresholds_subset.hpp"


PartialIndexVector::PartialIndexVector(uint32 numElements)
    : PartialIndexVector(numElements, false) {

}

PartialIndexVector::PartialIndexVector(uint32 numElements, bool init)
    : vector_(DenseVector<uint32>(numElements, init)) {

}

bool PartialIndexVector::isPartial() const {
    return true;
}

uint32 PartialIndexVector::getNumElements() const {
    return vector_.getNumElements();
}

void PartialIndexVector::setNumElements(uint32 numElements, bool freeMemory) {
    vector_.setNumElements(numElements, freeMemory);
}

uint32 PartialIndexVector::getIndex(uint32 pos) const {
    return vector_[pos];
}

PartialIndexVector::iterator PartialIndexVector::begin() {
    return vector_.begin();
}

PartialIndexVector::iterator PartialIndexVector::end() {
    return vector_.end();
}

PartialIndexVector::const_iterator PartialIndexVector::cbegin() const {
    return vector_.cbegin();
}

PartialIndexVector::const_iterator PartialIndexVector::cend() const {
    return vector_.cend();
}

std::unique_ptr<IStatisticsSubset> PartialIndexVector::createSubset(const IImmutableStatistics& statistics) const {
    return statistics.createSubset(*this);
}

std::unique_ptr<IRuleRefinement> PartialIndexVector::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                          uint32 featureIndex) const {
    return thresholdsSubset.createRuleRefinement(*this, featureIndex);
}
