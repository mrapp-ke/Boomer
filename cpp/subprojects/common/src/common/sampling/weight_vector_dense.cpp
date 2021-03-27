#include "common/sampling/weight_vector_dense.hpp"


DenseWeightVector::DenseWeightVector(uint32 numElements, uint32 sumOfWeights)
    : vector_(DenseVector<uint32>(numElements, true)), sumOfWeights_(sumOfWeights) {

}

DenseWeightVector::iterator DenseWeightVector::begin() {
    return vector_.begin();
}

DenseWeightVector::iterator DenseWeightVector::end() {
    return vector_.end();
}

DenseWeightVector::const_iterator DenseWeightVector::cbegin() const {
    return vector_.cbegin();
}

DenseWeightVector::const_iterator DenseWeightVector::cend() const {
    return vector_.cend();
}

bool DenseWeightVector::hasZeroWeights() const {
    return true;
}

uint32 DenseWeightVector::getWeight(uint32 pos) const {
    return vector_.getValue(pos);
}

uint32 DenseWeightVector::getSumOfWeights() const {
    return sumOfWeights_;
}
