#include "common/input/nominal_feature_mask_mixed.hpp"
#include "common/data/vector_bit.hpp"


/**
 * An implementation of the type `IMixedNominalFeatureMask` that uses a `BitVector` to store whether individual features
 * are nominal or not.
 */
class BitNominalFeatureMask final : public IMixedNominalFeatureMask {

    private:

        BitVector vector_;

    public:

        /**
         * @param numFeatures The total number of available features
         */
        BitNominalFeatureMask(uint32 numFeatures)
            : vector_(BitVector(numFeatures, true)) {

        }

        bool isNominal(uint32 featureIndex) const override {
            return vector_[featureIndex];
        }

        void setNominal(uint32 featureIndex, bool nominal) override {
            vector_.set(featureIndex, nominal);
        }

};

std::unique_ptr<IMixedNominalFeatureMask> createMixedNominalFeatureMask(uint32 numFeatures) {
    return std::make_unique<BitNominalFeatureMask>(numFeatures);
}
