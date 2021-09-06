#include "common/sampling/feature_sampling_random.hpp"
#include "common/indices/index_iterator.hpp"
#include "index_sampling.hpp"
#include <cmath>


RandomFeatureSubsetSelection::RandomFeatureSubsetSelection(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IIndexVector> RandomFeatureSubsetSelection::subSample(uint32 numFeatures, RNG& rng) const {
    uint32 numSamples;

    if (sampleSize_ > 0) {
            numSamples = (uint32) (sampleSize_ * numFeatures);
    } else {
            numSamples = (uint32) (log2(numFeatures - 1) + 1);
    }

    return sampleIndicesWithoutReplacement<IndexIterator>(IndexIterator(numFeatures), numFeatures, numSamples, rng);
}
