#include "common/sampling/partition_sampling_bi_random.hpp"

#include "common/sampling/partition_bi.hpp"
#include "common/util/validation.hpp"
#include "index_sampling.hpp"

/**
 * Allows to randomly split the training examples into two mutually exclusive sets that may be used as a training set
 * and a holdout set.
 */
class RandomBiPartitionSampling final : public IPartitionSampling {
    private:

        BiPartition partition_;

    public:

        /**
         * @param numTraining   The number of examples to be included in the training set
         * @param numHoldout    The number of examples to be included in the holdout set
         */
        RandomBiPartitionSampling(uint32 numTraining, uint32 numHoldout)
            : partition_(BiPartition(numTraining, numHoldout)) {}

        IPartition& partition(RNG& rng) override {
            uint32 numTraining = partition_.getNumFirst();
            uint32 numHoldout = partition_.getNumSecond();
            BiPartition::iterator trainingIterator = partition_.first_begin();
            setArrayToIncreasingValues<uint32>(trainingIterator, numTraining, 0, 1);
            BiPartition::iterator holdoutIterator = partition_.second_begin();

            for (uint32 i = 0; i < numHoldout; i++) {
                holdoutIterator[i] = numTraining + i;
            }

            uint32 numTotal = partition_.getNumElements();
            randomPermutation<BiPartition::iterator, BiPartition::iterator>(trainingIterator, holdoutIterator,
                                                                            numTraining, numTotal, numTraining, rng);
            return partition_;
        }
};

/**
 * Allows to create objects of the type `IPartitionSampling` that randomly split the training examples into two mutually
 * exclusive sets that may be used as a training set and a holdout set.
 */
class RandomBiPartitionSamplingFactory final : public IPartitionSamplingFactory {
    private:

        const float32 holdoutSetSize_;

    public:

        /**
         * @param holdoutSetSize The fraction of examples to be included in the holdout set (e.g. a value of 0.6
         *                       corresponds to 60 % of the available examples). Must be in (0, 1)
         */
        RandomBiPartitionSamplingFactory(float32 holdoutSetSize) : holdoutSetSize_(holdoutSetSize) {}

        std::unique_ptr<IPartitionSampling> create(const CContiguousLabelMatrix& labelMatrix) const override {
            uint32 numExamples = labelMatrix.getNumRows();
            uint32 numHoldout = (uint32) (holdoutSetSize_ * numExamples);
            uint32 numTraining = numExamples - numHoldout;
            return std::make_unique<RandomBiPartitionSampling>(numTraining, numHoldout);
        }

        std::unique_ptr<IPartitionSampling> create(const CsrLabelMatrix& labelMatrix) const override {
            uint32 numExamples = labelMatrix.getNumRows();
            uint32 numHoldout = (uint32) (holdoutSetSize_ * numExamples);
            uint32 numTraining = numExamples - numHoldout;
            return std::make_unique<RandomBiPartitionSampling>(numTraining, numHoldout);
        }
};

RandomBiPartitionSamplingConfig::RandomBiPartitionSamplingConfig() : holdoutSetSize_(0.33f) {}

float32 RandomBiPartitionSamplingConfig::getHoldoutSetSize() const {
    return holdoutSetSize_;
}

IRandomBiPartitionSamplingConfig& RandomBiPartitionSamplingConfig::setHoldoutSetSize(float32 holdoutSetSize) {
    assertGreater<float32>("holdoutSetSize", holdoutSetSize, 0);
    assertLess<float32>("holdoutSetSize", holdoutSetSize, 1);
    holdoutSetSize_ = holdoutSetSize;
    return *this;
}

std::unique_ptr<IPartitionSamplingFactory> RandomBiPartitionSamplingConfig::createPartitionSamplingFactory() const {
    return std::make_unique<RandomBiPartitionSamplingFactory>(holdoutSetSize_);
}
