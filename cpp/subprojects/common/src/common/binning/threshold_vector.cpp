#include "common/binning/threshold_vector.hpp"


ThresholdVector::ThresholdVector(MissingFeatureVector& missingFeatureVector, uint32 numElements)
    : ThresholdVector(missingFeatureVector, numElements, false) {

}

ThresholdVector::ThresholdVector(MissingFeatureVector& missingFeatureVector, uint32 numElements, bool init)
    : MissingFeatureVector(missingFeatureVector), vector_(DenseVector<float32>(numElements, init)),
      sparseBinIndex_(numElements) {

}

ThresholdVector::iterator ThresholdVector::begin() {
    return vector_.begin();
}

ThresholdVector::iterator ThresholdVector::end() {
    return vector_.end();
}

ThresholdVector::const_iterator ThresholdVector::cbegin() const {
    return vector_.cbegin();
}

ThresholdVector::const_iterator ThresholdVector::cend() const {
    return vector_.cend();
}

uint32 ThresholdVector::getNumElements() const {
    return vector_.getNumElements();
}

void ThresholdVector::setNumElements(uint32 numElements, bool freeMemory) {
    vector_.setNumElements(numElements, freeMemory);

    if (sparseBinIndex_ > numElements) {
        sparseBinIndex_ = numElements;
    }
}

uint32 ThresholdVector::getSparseBinIndex() const {
    return sparseBinIndex_;
}

void ThresholdVector::setSparseBinIndex(uint32 sparseBinIndex) {
    uint32 numElements = this->getNumElements();

    if (sparseBinIndex > numElements) {
        sparseBinIndex_ = numElements;
    } else {
        sparseBinIndex_ = sparseBinIndex;
    }
}
