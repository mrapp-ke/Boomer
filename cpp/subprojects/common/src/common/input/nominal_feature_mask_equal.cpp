#include "common/input/nominal_feature_mask_equal.hpp"


/**
 * An implementation of the type `IEqualNominalFeatureMask` that stores whether all features are nominal or not.
 */
class EqualNominalFeatureMask final : public IEqualNominalFeatureMask {

    private:

        bool nominal_;

    public:

        /**
         * @param nominal True, if all features are nominal, false, if all features are numerical/ordinal
         */
        EqualNominalFeatureMask(bool nominal)
            : nominal_(nominal) {

        }

        bool isNominal(uint32 featureIndex) const override {
            return nominal_;
        }

};

std::unique_ptr<IEqualNominalFeatureMask> createEqualNominalFeatureMask(bool nominal) {
    return std::make_unique<EqualNominalFeatureMask>(nominal);
}
