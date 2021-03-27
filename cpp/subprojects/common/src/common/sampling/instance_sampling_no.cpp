#include "common/sampling/instance_sampling_no.hpp"
#include "common/sampling/weight_vector_equal.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"


std::unique_ptr<IWeightVector> NoInstanceSubSampling::subSample(const SinglePartition& partition, RNG& rng) const {
    return std::make_unique<EqualWeightVector>(partition.getNumElements());
}

std::unique_ptr<IWeightVector> NoInstanceSubSampling::subSample(const BiPartition& partition, RNG& rng) const {
    uint32 numExamples = partition.getNumElements();
    uint32 numTrainingExamples = partition.getNumFirst();
    BiPartition::const_iterator indexIterator = partition.first_cbegin();
    std::unique_ptr<DenseWeightVector> weightVectorPtr = std::make_unique<DenseWeightVector>(numExamples,
                                                                                             numTrainingExamples);
    DenseWeightVector::iterator weightIterator = weightVectorPtr->begin();

    for (uint32 i = 0; i < numTrainingExamples; i++) {
        uint32 index = indexIterator[i];
        weightIterator[index] = 1;
    }

    return weightVectorPtr;
}
