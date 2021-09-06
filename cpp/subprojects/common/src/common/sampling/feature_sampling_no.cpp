#include "common/sampling/feature_sampling_no.hpp"
#include "common/indices/index_vector_complete.hpp"


/**
 * An implementation of the class `IFeatureSampling` that does not perform any sampling, but includes all features.
 */
class NoFeatureSampling final : public IFeatureSampling {

    private:

        CompleteIndexVector indexVector_;

    public:

        /**
         * @param numFeatures The total number of available features
         */
        NoFeatureSampling(uint32 numFeatures)
            : indexVector_(CompleteIndexVector(numFeatures)) {

        }

        const IIndexVector& sample(RNG& rng) override {
            return indexVector_;
        }

};

std::unique_ptr<IFeatureSampling> NoFeatureSamplingFactory::create(uint32 numFeatures) const {
    return std::make_unique<NoFeatureSampling>(numFeatures);
}
