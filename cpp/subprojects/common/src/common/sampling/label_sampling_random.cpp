#include "common/sampling/label_sampling_random.hpp"
#include "common/indices/index_iterator.hpp"
#include "index_sampling.hpp"

RandomLabelSubsetSelection::RandomLabelSubsetSelection(uint32 numSamples)
    : numSamples_(numSamples) {

}

std::unique_ptr<IIndexVector> RandomLabelSubsetSelection::subSample(uint32 numLabels, RNG& rng) const {
    return sampleIndicesWithoutReplacement<IndexIterator>(IndexIterator(numLabels), numLabels, numSamples_, rng);
}
