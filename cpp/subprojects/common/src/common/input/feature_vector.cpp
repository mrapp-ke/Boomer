#include "common/input/feature_vector.hpp"


FeatureVector::FeatureVector(uint32 numElements)
    : vector_(SparseArrayVector<float32>(numElements)) {

}

FeatureVector::iterator FeatureVector::begin() {
    return vector_.begin();
}

FeatureVector::iterator FeatureVector::end() {
    return vector_.end();
}

FeatureVector::const_iterator FeatureVector::cbegin() const {
    return vector_.cbegin();
}

FeatureVector::const_iterator FeatureVector::cend() const {
    return vector_.cend();
}

uint32 FeatureVector::getNumElements() const {
    return vector_.getNumElements();
}

void FeatureVector::setNumElements(uint32 numElements, bool freeMemory) {
    return vector_.setNumElements(numElements, freeMemory);
}

void FeatureVector::sortByValues() {
    vector_.sortByValues();
}
