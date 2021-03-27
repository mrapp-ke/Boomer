#include "common/sampling/instance_sampling_random.hpp"
#include "common/indices/index_iterator.hpp"
#include "weight_sampling.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"


RandomInstanceSubsetSelection::RandomInstanceSubsetSelection(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IWeightVector> RandomInstanceSubsetSelection::subSample(const SinglePartition& partition,
                                                                        RNG& rng) const {
    uint32 numExamples = partition.getNumElements();
    uint32 numSamples = (uint32) (sampleSize_ * numExamples);
    return sampleWeightsWithoutReplacement<IndexIterator>(IndexIterator(numExamples), numExamples, numSamples,
                                                          numExamples, rng);
}

std::unique_ptr<IWeightVector> RandomInstanceSubsetSelection::subSample(const BiPartition& partition, RNG& rng) const {
    uint32 numExamples = partition.getNumElements();
    uint32 numTrainingExamples = partition.getNumFirst();
    uint32 numSamples = (uint32) (sampleSize_ * numTrainingExamples);
    return sampleWeightsWithoutReplacement<BiPartition::const_iterator>(partition.first_cbegin(), numTrainingExamples,
                                                                        numSamples, numExamples, rng);
}
