#include "common/sampling/instance_sampling_bagging.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"


Bagging::Bagging(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IWeightVector> Bagging::subSample(const SinglePartition& partition, RNG& rng) const {
    uint32 numExamples = partition.getNumElements();
    uint32 numSamples = (uint32) (sampleSize_ * numExamples);
    std::unique_ptr<DenseWeightVector> weightVectorPtr = std::make_unique<DenseWeightVector>(numExamples, numSamples);
    DenseWeightVector::iterator weightIterator = weightVectorPtr->begin();

    for (uint32 i = 0; i < numSamples; i++) {
        // Randomly select the index of an example...
        uint32 randomIndex = rng.random(0, numExamples);

        // Update weight at the selected index...
        weightIterator[randomIndex] += 1;
    }

    return weightVectorPtr;
}

std::unique_ptr<IWeightVector> Bagging::subSample(const BiPartition& partition, RNG& rng) const {
    uint32 numExamples = partition.getNumElements();
    uint32 numTrainingExamples = partition.getNumFirst();
    uint32 numSamples = (uint32) (sampleSize_ * numTrainingExamples);
    BiPartition::const_iterator indexIterator = partition.first_cbegin();
    std::unique_ptr<DenseWeightVector> weightVectorPtr = std::make_unique<DenseWeightVector>(numExamples, numSamples);
    DenseWeightVector::iterator weightIterator = weightVectorPtr->begin();

    for (uint32 i = 0; i < numSamples; i++) {
        // Randomly select the index of an example...
        uint32 randomIndex = rng.random(0, numTrainingExamples);
        uint32 sampledIndex = indexIterator[randomIndex];

        // Update weight at the selected index...
        weightIterator[sampledIndex] += 1;
    }

    return weightVectorPtr;
}
