#include "common/sampling/partition_bi.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/thresholds/thresholds_subset.hpp"
#include "common/rule_refinement/refinement.hpp"
#include "common/head_refinement/prediction.hpp"


static inline std::unordered_set<uint32>* createSet(BiPartition::const_iterator iterator, uint32 numElements) {
    std::unordered_set<uint32>* set = new std::unordered_set<uint32>();

    for (uint32 i = 0; i < numElements; i++) {
        uint32 index = iterator[i];
        set->insert(index);
    }

    return set;
}

BiPartition::BiPartition(uint32 numFirst, uint32 numSecond)
    : vector_(DenseVector<uint32>(numFirst + numSecond)), numFirst_(numFirst), firstSet_(nullptr), secondSet_(nullptr) {

}

BiPartition::~BiPartition() {
    delete firstSet_;
    delete secondSet_;
}

BiPartition::iterator BiPartition::first_begin() {
    return vector_.begin();
}

BiPartition::iterator BiPartition::first_end() {
    return &vector_.begin()[numFirst_];
}

BiPartition::const_iterator BiPartition::first_cbegin() const {
    return vector_.cbegin();
}

BiPartition::const_iterator BiPartition::first_cend() const {
    return &vector_.cbegin()[numFirst_];
}

BiPartition::iterator BiPartition::second_begin() {
    return &vector_.begin()[numFirst_];
}

BiPartition::iterator BiPartition::second_end() {
    return vector_.end();
}

BiPartition::const_iterator BiPartition::second_cbegin() const {
    return &vector_.cbegin()[numFirst_];
}

BiPartition::const_iterator BiPartition::second_cend() const {
    return vector_.cend();
}

uint32 BiPartition::getNumFirst() const {
    return numFirst_;
}

uint32 BiPartition::getNumSecond() const {
    return vector_.getNumElements() - numFirst_;
}

uint32 BiPartition::getNumElements() const {
    return vector_.getNumElements();
}

const std::unordered_set<uint32>& BiPartition::getFirstSet() {
    if (firstSet_ == nullptr) {
        firstSet_ = createSet(this->first_cbegin(), this->getNumFirst());
    }

    return *firstSet_;
}

const std::unordered_set<uint32>& BiPartition::getSecondSet() {
    if (secondSet_ == nullptr) {
        secondSet_ = createSet(this->second_cbegin(), this->getNumSecond());
    }

    return *secondSet_;
}

std::unique_ptr<IWeightVector> BiPartition::subSample(const IInstanceSubSampling& instanceSubSampling, RNG& rng) const {
    return instanceSubSampling.subSample(*this, rng);
}

float64 BiPartition::evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const ICoverageState& coverageState,
                                         const AbstractPrediction& head) {
    return coverageState.evaluateOutOfSample(thresholdsSubset, *this, head);
}

void BiPartition::recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const ICoverageState& coverageState,
                                        Refinement& refinement) {
    coverageState.recalculatePrediction(thresholdsSubset, *this, refinement);
}
