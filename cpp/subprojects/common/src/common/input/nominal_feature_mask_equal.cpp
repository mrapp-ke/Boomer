#include "common/input/nominal_feature_mask_equal.hpp"


EqualNominalFeatureMask::EqualNominalFeatureMask(bool nominal)
    : nominal_(nominal) {

}

bool EqualNominalFeatureMask::isNominal(uint32 featureIndex) const {
    return nominal_;
}
