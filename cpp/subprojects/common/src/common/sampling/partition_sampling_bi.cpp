#include "common/sampling/partition_sampling_bi.hpp"
#include "common/sampling/partition_bi.hpp"
#include "index_sampling.hpp"


BiPartitionSampling::BiPartitionSampling(float32 holdoutSetSize)
    : holdoutSetSize_(holdoutSetSize) {

}

std::unique_ptr<IPartition> BiPartitionSampling::partition(uint32 numExamples, RNG& rng) const {
    uint32 numHoldout = (uint32) (holdoutSetSize_ * numExamples);
    uint32 numTraining = numExamples - numHoldout;
    std::unique_ptr<BiPartition> partitionPtr = std::make_unique<BiPartition>(numTraining, numHoldout);
    BiPartition::iterator trainingIterator = partitionPtr->first_begin();
    BiPartition::iterator holdoutIterator = partitionPtr->second_begin();

    for (uint32 i = 0; i < numTraining; i++) {
        trainingIterator[i] = i;
    }

    for (uint32 i = 0; i < numHoldout; i++) {
        holdoutIterator[i] = numTraining + i;
    }

    randomPermutation<BiPartition::iterator, BiPartition::iterator>(trainingIterator, holdoutIterator, numTraining,
                                                                    numExamples, rng);
    return partitionPtr;
}
