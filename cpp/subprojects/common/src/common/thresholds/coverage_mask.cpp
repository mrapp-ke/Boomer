#include "common/thresholds/coverage_mask.hpp"
#include "common/thresholds/thresholds_subset.hpp"
#include "common/rule_refinement/refinement.hpp"
#include "common/head_refinement/prediction.hpp"
#include "common/data/arrays.hpp"


CoverageMask::CoverageMask(uint32 numElements)
    : array_(new uint32[numElements]{0}), numElements_(numElements), target_(0) {

}

CoverageMask::CoverageMask(const CoverageMask& coverageMask)
    : array_(new uint32[coverageMask.numElements_]), numElements_(coverageMask.numElements_),
      target_(coverageMask.target_) {
    copyArray(coverageMask.array_, array_, numElements_);
}

CoverageMask::~CoverageMask() {
    delete[] array_;
}

CoverageMask::iterator CoverageMask::begin() {
    return array_;
}

CoverageMask::iterator CoverageMask::end() {
    return &array_[numElements_];
}

CoverageMask::const_iterator CoverageMask::cbegin() const {
    return array_;
}

CoverageMask::const_iterator CoverageMask::cend() const {
    return &array_[numElements_];
}

uint32 CoverageMask::getNumElements() const {
    return numElements_;
}

uint32 CoverageMask::getTarget() const {
    return target_;
}

void CoverageMask::setTarget(uint32 target) {
    target_ = target;
}

void CoverageMask::reset() {
    target_ = 0;
    setArrayToZeros(array_, numElements_);
}

bool CoverageMask::isCovered(uint32 pos) const {
    return array_[pos] == target_;
}

std::unique_ptr<ICoverageState> CoverageMask::copy() const {
    return std::make_unique<CoverageMask>(*this);
}

float64 CoverageMask::evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                          const AbstractPrediction& head) const {
    return thresholdsSubset.evaluateOutOfSample(partition, *this, head);
}

float64 CoverageMask::evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                          const AbstractPrediction& head) const {
    return thresholdsSubset.evaluateOutOfSample(partition, *this, head);
}

void CoverageMask::recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                         Refinement& refinement) const {
    thresholdsSubset.recalculatePrediction(partition, *this, refinement);
}

void CoverageMask::recalculatePrediction(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                         Refinement& refinement) const {
    thresholdsSubset.recalculatePrediction(partition, *this, refinement);
}
