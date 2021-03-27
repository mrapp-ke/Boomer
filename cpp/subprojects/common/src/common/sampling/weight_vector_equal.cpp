#include "common/sampling/weight_vector_equal.hpp"


EqualWeightVector::EqualWeightVector(uint32 numElements)
    : numElements_(numElements) {

}

bool EqualWeightVector::hasZeroWeights() const {
    return false;
}

uint32 EqualWeightVector::getWeight(uint32 pos) const {
    return 1;
}

uint32 EqualWeightVector::getSumOfWeights() const {
    return numElements_;
}
