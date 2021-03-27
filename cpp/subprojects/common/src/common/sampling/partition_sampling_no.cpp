#include "common/sampling/partition_sampling_no.hpp"
#include "common/sampling/partition_single.hpp"


std::unique_ptr<IPartition> NoPartitionSampling::partition(uint32 numExamples, RNG& rng) const {
    return std::make_unique<SinglePartition>(numExamples);
}
