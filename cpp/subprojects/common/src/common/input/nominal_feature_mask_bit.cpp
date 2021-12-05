#include "common/input/nominal_feature_mask_bit.hpp"


BitNominalFeatureMask::BitNominalFeatureMask(uint32 numFeatures)
    : vector_(BitVector(numFeatures, true)) {

}

bool BitNominalFeatureMask::isNominal(uint32 featureIndex) const {
    return vector_[featureIndex];
}

void BitNominalFeatureMask::setNominal(uint32 featureIndex) {
    vector_.set(featureIndex, true);
}
