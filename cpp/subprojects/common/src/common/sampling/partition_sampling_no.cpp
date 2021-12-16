#include "common/sampling/partition_sampling_no.hpp"
#include "common/sampling/partition_single.hpp"


/**
 * An implementation of the class `IPartitionSampling` that does not split the training examples, but includes all of
 * them in the training set.
 */
class NoPartitionSampling final : public IPartitionSampling {

    private:

        SinglePartition partition_;

    public:

        /**
         * @param numExamples The total number of available training examples
         */
        NoPartitionSampling(uint32 numExamples)
            : partition_(SinglePartition(numExamples)) {

        }

        IPartition& partition(RNG& rng) override {
            return partition_;
        }

};

std::unique_ptr<IPartitionSampling> NoPartitionSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix) const {
    return std::make_unique<NoPartitionSampling>(labelMatrix.getNumRows());
}

std::unique_ptr<IPartitionSampling> NoPartitionSamplingFactory::create(const CsrLabelMatrix& labelMatrix) const {
    return std::make_unique<NoPartitionSampling>(labelMatrix.getNumRows());
}
