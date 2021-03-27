#include "common/indices/index_vector_full.hpp"
#include "common/head_refinement/head_refinement.hpp"
#include "common/head_refinement/head_refinement_factory.hpp"
#include "common/statistics/statistics_immutable.hpp"
#include "common/thresholds/thresholds_subset.hpp"


FullIndexVector::FullIndexVector(uint32 numElements) {
    numElements_ = numElements;
}

bool FullIndexVector::isPartial() const {
    return false;
}

uint32 FullIndexVector::getNumElements() const {
    return numElements_;
}

void FullIndexVector::setNumElements(uint32 numElements, bool freeMemory) {
    numElements_ = numElements;
}

uint32 FullIndexVector::getIndex(uint32 pos) const {
    return pos;
}

FullIndexVector::const_iterator FullIndexVector::cbegin() const {
    return IndexIterator();
}

FullIndexVector::const_iterator FullIndexVector::cend() const {
    return IndexIterator(numElements_);
}

std::unique_ptr<IStatisticsSubset> FullIndexVector::createSubset(const IImmutableStatistics& statistics) const {
    return statistics.createSubset(*this);
}

std::unique_ptr<IRuleRefinement> FullIndexVector::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                        uint32 featureIndex) const {
    return thresholdsSubset.createRuleRefinement(*this, featureIndex);
}

std::unique_ptr<IHeadRefinement> FullIndexVector::createHeadRefinement(const IHeadRefinementFactory& factory) const {
    return factory.create(*this);
}
