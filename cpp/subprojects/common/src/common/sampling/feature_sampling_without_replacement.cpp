#include "common/sampling/feature_sampling_without_replacement.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/iterator/index_iterator.hpp"
#include "common/util/validation.hpp"
#include "index_sampling.hpp"
#include <cmath>


/**
 * Allows to select a subset of the available features without replacement.
 */
class FeatureSamplingWithoutReplacement final : public IFeatureSampling {

    private:

        uint32 numFeatures_;

        PartialIndexVector indexVector_;

    public:

        /**
         * @param numFeatures   The total number of available features
         * @param numSamples    The number of features to be included in the sample
         */
        FeatureSamplingWithoutReplacement(uint32 numFeatures, uint32 numSamples)
            : numFeatures_(numFeatures), indexVector_(PartialIndexVector(numSamples)) {

        }

        const IIndexVector& sample(RNG& rng) override {
            sampleIndicesWithoutReplacement<IndexIterator>(indexVector_, IndexIterator(numFeatures_), numFeatures_,
                                                           rng);
            return indexVector_;
        }

};

/**
 * Allows to create instances of the type `IFeatureSampling` that select a random subset of the available features
 * without replacement.
 */
class FeatureSamplingWithoutReplacementFactory final : public IFeatureSamplingFactory {

    private:

        uint32 numFeatures_;

        uint32 numSamples_;

    public:

        /**
         * @param numFeatures   The total number of available features
         * @param numSamples    The number of features to be included in the sample
         */
        FeatureSamplingWithoutReplacementFactory(uint32 numFeatures, uint32 numSamples)
            : numFeatures_(numFeatures), numSamples_(numSamples) {

        }

        std::unique_ptr<IFeatureSampling> create() const override {
            return std::make_unique<FeatureSamplingWithoutReplacement>(numFeatures_, numSamples_);
        }

};

FeatureSamplingWithoutReplacementConfig::FeatureSamplingWithoutReplacementConfig()
    : sampleSize_(0) {

}

float32 FeatureSamplingWithoutReplacementConfig::getSampleSize() const {
    return sampleSize_;
}

IFeatureSamplingWithoutReplacementConfig& FeatureSamplingWithoutReplacementConfig::setSampleSize(float32 sampleSize) {
    assertGreaterOrEqual<float32>("sampleSize", sampleSize, 0);
    assertLess<float32>("sampleSize", sampleSize, 1);
    sampleSize_ = sampleSize;
    return *this;
}

std::unique_ptr<IFeatureSamplingFactory> FeatureSamplingWithoutReplacementConfig::createFeatureSamplingFactory(
        const IFeatureMatrix& featureMatrix) const {
    uint32 numFeatures = featureMatrix.getNumCols();
    uint32 numSamples = (uint32) (sampleSize_ > 0 ? sampleSize_ * numFeatures : log2(numFeatures - 1) + 1);
    return std::make_unique<FeatureSamplingWithoutReplacementFactory>(numFeatures, numSamples);
}
