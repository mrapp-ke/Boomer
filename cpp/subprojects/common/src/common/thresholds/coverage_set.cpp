#include "common/thresholds/coverage_set.hpp"
#include "common/thresholds/thresholds_subset.hpp"
#include "common/rule_refinement/refinement.hpp"
#include "common/head_refinement/prediction.hpp"
#include "common/data/arrays.hpp"


CoverageSet::CoverageSet(uint32 numElements)
    : array_(new uint32[numElements]), numElements_(numElements), numCovered_(numElements) {
    for (uint32 i = 0; i < numElements; i++) {
        array_[i] = i;
    }
}

CoverageSet::CoverageSet(const CoverageSet& coverageSet)
    : array_(new uint32[coverageSet.numElements_]), numElements_(coverageSet.numElements_),
      numCovered_(coverageSet.numCovered_) {
    copyArray(coverageSet.array_, array_, numCovered_);
}

CoverageSet::~CoverageSet() {
    delete[] array_;
}

CoverageSet::iterator CoverageSet::begin() {
    return array_;
}

CoverageSet::iterator CoverageSet::end() {
    return &array_[numCovered_];
}

CoverageSet::const_iterator CoverageSet::cbegin() const {
    return array_;
}

CoverageSet::const_iterator CoverageSet::cend() const {
    return &array_[numCovered_];
}

uint32 CoverageSet::getNumElements() const {
    return numElements_;
}

uint32 CoverageSet::getNumCovered() const {
    return numCovered_;
}

void CoverageSet::setNumCovered(uint32 numCovered) {
    numCovered_ = numCovered;
}

void CoverageSet::reset() {
    numCovered_ = numElements_;

    for (uint32 i = 0; i < numElements_; i++) {
        array_[i] = i;
    }
}

std::unique_ptr<ICoverageState> CoverageSet::copy() const {
    return std::make_unique<CoverageSet>(*this);
}

float64 CoverageSet::evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                         const AbstractPrediction& head) const {
    return thresholdsSubset.evaluateOutOfSample(partition, *this, head);
}

float64 CoverageSet::evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                         const AbstractPrediction& head) const {
    return thresholdsSubset.evaluateOutOfSample(partition, *this, head);
}

void CoverageSet::recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                        Refinement& refinement) const {
    thresholdsSubset.recalculatePrediction(partition, *this, refinement);
}

void CoverageSet::recalculatePrediction(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                        Refinement& refinement) const {
    thresholdsSubset.recalculatePrediction(partition, *this, refinement);
}
