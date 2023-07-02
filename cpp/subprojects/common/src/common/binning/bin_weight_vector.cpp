#include "common/binning/bin_weight_vector.hpp"

#include "common/data/arrays.hpp"

BinWeightVector::BinWeightVector(uint32 numElements) : vector_(DenseVector<uint32>(numElements)) {}

void BinWeightVector::clear() {
    setArrayToZeros(vector_.begin(), vector_.getNumElements());
}

void BinWeightVector::increaseWeight(uint32 pos) {
    vector_[pos] += 1;
}

bool BinWeightVector::operator[](uint32 pos) const {
    return vector_[pos] != 0;
}

uint32 BinWeightVector::getNumElements() const {
    return vector_.getNumElements();
}
