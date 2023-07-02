#include "common/sampling/feature_sampling_without_replacement.hpp"

#include "common/indices/index_vector_partial.hpp"
#include "common/iterator/index_iterator.hpp"
#include "common/sampling/feature_sampling_predefined.hpp"
#include "common/util/validation.hpp"
#include "index_sampling.hpp"

/**
 * Allows to select a subset of the available features without replacement.
 */
class FeatureSamplingWithoutReplacement final : public IFeatureSampling {
    private:

        const uint32 numFeatures_;

        const uint32 numSamples_;

        const uint32 numRetained_;

        PartialIndexVector indexVector_;

    public:

        /**
         * @param numFeatures   The total number of available features
         * @param numSamples    The number of features to be included in the sample
         * @param numRetained   The number of trailing features to be always included in the sample
         */
        FeatureSamplingWithoutReplacement(uint32 numFeatures, uint32 numSamples, uint32 numRetained)
            : numFeatures_(numFeatures), numSamples_(numSamples), numRetained_(numRetained),
              indexVector_(PartialIndexVector(numSamples + numRetained)) {
            if (numRetained > 0) {
                PartialIndexVector::iterator iterator = indexVector_.begin();
                uint32 offset = numFeatures - numRetained;

                for (uint32 i = 0; i < numRetained; i++) {
                    iterator[i] = offset + i;
                }
            }
        }

        const IIndexVector& sample(RNG& rng) override {
            uint32 numTotal = numFeatures_ - numRetained_;
            sampleIndicesWithoutReplacement<IndexIterator>(&indexVector_.begin()[numRetained_], numSamples_,
                                                           IndexIterator(numTotal), numTotal, rng);
            return indexVector_;
        }

        std::unique_ptr<IFeatureSampling> createBeamSearchFeatureSampling(RNG& rng, bool resample) override {
            if (resample) {
                return std::make_unique<FeatureSamplingWithoutReplacement>(numFeatures_, numSamples_, numRetained_);
            } else {
                return std::make_unique<PredefinedFeatureSampling>(this->sample(rng));
            }
        }
};

/**
 * Allows to create instances of the type `IFeatureSampling` that select a random subset of the available features
 * without replacement.
 */
class FeatureSamplingWithoutReplacementFactory final : public IFeatureSamplingFactory {
    private:

        const uint32 numFeatures_;

        const uint32 numSamples_;

        const uint32 numRetained_;

    public:

        /**
         * @param numFeatures   The total number of available features
         * @param numSamples    The number of features to be included in the sample
         * @param numRetained   The number of trailing features to be always included in the sample
         */
        FeatureSamplingWithoutReplacementFactory(uint32 numFeatures, uint32 numSamples, uint32 numRetained)
            : numFeatures_(numFeatures), numSamples_(numSamples), numRetained_(numRetained) {}

        std::unique_ptr<IFeatureSampling> create() const override {
            return std::make_unique<FeatureSamplingWithoutReplacement>(numFeatures_, numSamples_, numRetained_);
        }
};

FeatureSamplingWithoutReplacementConfig::FeatureSamplingWithoutReplacementConfig() : sampleSize_(0), numRetained_(0) {}

float32 FeatureSamplingWithoutReplacementConfig::getSampleSize() const {
    return sampleSize_;
}

IFeatureSamplingWithoutReplacementConfig& FeatureSamplingWithoutReplacementConfig::setSampleSize(float32 sampleSize) {
    assertGreaterOrEqual<float32>("sampleSize", sampleSize, 0);
    assertLess<float32>("sampleSize", sampleSize, 1);
    sampleSize_ = sampleSize;
    return *this;
}

uint32 FeatureSamplingWithoutReplacementConfig::getNumRetained() const {
    return numRetained_;
}

IFeatureSamplingWithoutReplacementConfig& FeatureSamplingWithoutReplacementConfig::setNumRetained(uint32 numRetained) {
    assertGreaterOrEqual<uint32>("numRetained", numRetained, 0);
    numRetained_ = numRetained;
    return *this;
}

std::unique_ptr<IFeatureSamplingFactory> FeatureSamplingWithoutReplacementConfig::createFeatureSamplingFactory(
  const IFeatureMatrix& featureMatrix) const {
    uint32 numFeatures = featureMatrix.getNumCols();
    uint32 numRetained = std::min(numRetained_, numFeatures);
    uint32 numRemainingFeatures = numFeatures - numRetained;
    uint32 numSamples =
      (uint32) (sampleSize_ > 0 ? sampleSize_ * numRemainingFeatures : log2(numRemainingFeatures - 1) + 1);
    return std::make_unique<FeatureSamplingWithoutReplacementFactory>(numFeatures, numSamples, numRetained);
}

bool FeatureSamplingWithoutReplacementConfig::isSamplingUsed() const {
    return true;
}
