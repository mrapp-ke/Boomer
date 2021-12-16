#include "common/sampling/weight_vector_bit.hpp"


BitWeightVector::BitWeightVector(uint32 numElements)
    : BitWeightVector(numElements, false) {

}

BitWeightVector::BitWeightVector(uint32 numElements, bool init)
    : vector_(BitVector(numElements, init)), numNonZeroWeights_(0) {

}

uint32 BitWeightVector::getNumElements() const {
    return vector_.getNumElements();
}

uint32 BitWeightVector::getNumNonZeroWeights() const {
    return numNonZeroWeights_;
}

void BitWeightVector::setNumNonZeroWeights(uint32 numNonZeroWeights) {
    numNonZeroWeights_ = numNonZeroWeights;
}

bool BitWeightVector::hasZeroWeights() const {
    return numNonZeroWeights_ < vector_.getNumElements();
}

void BitWeightVector::set(uint32 pos, bool weight) {
    vector_.set(pos, weight);
}

void BitWeightVector::clear() {
    vector_.clear();
}

float64 BitWeightVector::getWeight(uint32 pos) const {
    return (float64) vector_[pos];
}
