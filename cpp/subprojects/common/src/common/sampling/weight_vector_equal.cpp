#include "common/sampling/weight_vector_equal.hpp"


EqualWeightVector::EqualWeightVector(uint32 numElements)
    : numElements_(numElements) {

}

uint32 EqualWeightVector::getNumElements() const {
    return numElements_;
}

uint32 EqualWeightVector::getNumNonZeroWeights() const {
    return numElements_;
}

bool EqualWeightVector::hasZeroWeights() const {
    return false;
}

float64 EqualWeightVector::getWeight(uint32 pos) const {
    return 1;
}
