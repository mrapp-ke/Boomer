#include "common/input/feature_info_mixed.hpp"

#include "common/data/vector_bit.hpp"
#include "common/input/feature_type_nominal.hpp"
#include "common/input/feature_type_numerical.hpp"
#include "common/input/feature_type_ordinal.hpp"

/**
 * An implementation of the type `IMixedFeatureInfo` that uses `BitVector`s to store whether individual features are
 * ordinal, nominal or numerical.
 */
class BitFeatureInfo final : public IMixedFeatureInfo {
    private:

        BitVector ordinalBitVector_;

        BitVector nominalBitVector_;

    public:

        /**
         * @param numFeatures The total number of available features
         */
        BitFeatureInfo(uint32 numFeatures)
            : ordinalBitVector_(BitVector(numFeatures, true)), nominalBitVector_(BitVector(numFeatures, true)) {}

        std::unique_ptr<IFeatureType> createFeatureType(uint32 featureIndex) const override {
            if (ordinalBitVector_[featureIndex]) {
                return std::make_unique<OrdinalFeatureType>();
            } else if (nominalBitVector_[featureIndex]) {
                return std::make_unique<NominalFeatureType>();
            } else {
                return std::make_unique<NumericalFeatureType>();
            }
        }

        void setNumerical(uint32 featureIndex) override {
            ordinalBitVector_.set(featureIndex, false);
            nominalBitVector_.set(featureIndex, false);
        }

        void setOrdinal(uint32 featureIndex) override {
            ordinalBitVector_.set(featureIndex, true);
            nominalBitVector_.set(featureIndex, false);
        }

        void setNominal(uint32 featureIndex) override {
            ordinalBitVector_.set(featureIndex, false);
            nominalBitVector_.set(featureIndex, true);
        }
};

std::unique_ptr<IMixedFeatureInfo> createMixedFeatureInfo(uint32 numFeatures) {
    return std::make_unique<BitFeatureInfo>(numFeatures);
}
