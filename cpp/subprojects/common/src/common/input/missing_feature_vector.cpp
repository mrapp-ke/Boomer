#include "common/input/missing_feature_vector.hpp"


MissingFeatureVector::MissingFeatureVector()
    : missingIndicesPtr_(std::make_unique<BinaryDokVector>()) {

}

MissingFeatureVector::MissingFeatureVector(MissingFeatureVector& missingFeatureVector)
    : missingIndicesPtr_(std::move(missingFeatureVector.missingIndicesPtr_)) {

}

MissingFeatureVector::missing_index_const_iterator MissingFeatureVector::missing_indices_cbegin() const {
    return missingIndicesPtr_->indices_cbegin();
}

MissingFeatureVector::missing_index_const_iterator MissingFeatureVector::missing_indices_cend() const {
    return missingIndicesPtr_->indices_cend();
}

void MissingFeatureVector::addMissingIndex(uint32 index) {
    missingIndicesPtr_->setValue(index);
}

bool MissingFeatureVector::isMissing(uint32 index) const {
    return missingIndicesPtr_->getValue(index);
}

void MissingFeatureVector::clearMissingIndices() {
    missingIndicesPtr_->setAllToZero();
}
