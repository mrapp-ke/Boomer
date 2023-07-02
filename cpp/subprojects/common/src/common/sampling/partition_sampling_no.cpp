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
        NoPartitionSampling(uint32 numExamples) : partition_(SinglePartition(numExamples)) {}

        IPartition& partition(RNG& rng) override {
            return partition_;
        }
};

/**
 * Allows to create objects of the type `IPartitionSampling` that do not split the training examples, but include all of
 * them in the training set.
 */
class NoPartitionSamplingFactory final : public IPartitionSamplingFactory {
    public:

        std::unique_ptr<IPartitionSampling> create(const CContiguousLabelMatrix& labelMatrix) const override {
            return std::make_unique<NoPartitionSampling>(labelMatrix.getNumRows());
        }

        std::unique_ptr<IPartitionSampling> create(const CsrLabelMatrix& labelMatrix) const override {
            return std::make_unique<NoPartitionSampling>(labelMatrix.getNumRows());
        }
};

std::unique_ptr<IPartitionSamplingFactory> NoPartitionSamplingConfig::createPartitionSamplingFactory() const {
    return std::make_unique<NoPartitionSamplingFactory>();
}
