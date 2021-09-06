#include "common/sampling/partition_sampling_bi_random.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/validation.hpp"
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
            : partition_(BiPartition(numTraining, numHoldout)) {

        }

        IPartition& partition(RNG& rng) override {
            uint32 numTraining = partition_.getNumFirst();
            uint32 numHoldout = partition_.getNumSecond();
            BiPartition::iterator trainingIterator = partition_.first_begin();
            BiPartition::iterator holdoutIterator = partition_.second_begin();

            for (uint32 i = 0; i < numTraining; i++) {
                trainingIterator[i] = i;
            }

            for (uint32 i = 0; i < numHoldout; i++) {
                holdoutIterator[i] = numTraining + i;
            }

            uint32 numTotal = partition_.getNumElements();
            randomPermutation<BiPartition::iterator, BiPartition::iterator>(trainingIterator, holdoutIterator,
                                                                            numTraining, numTotal, numTraining, rng);
            return partition_;
        }

};

RandomBiPartitionSamplingFactory::RandomBiPartitionSamplingFactory(float32 holdoutSetSize)
    : holdoutSetSize_(holdoutSetSize) {
    assertGreater<float32>("holdoutSetSize", holdoutSetSize, 0);
    assertLess<float32>("holdoutSetSize", holdoutSetSize, 1);
}

std::unique_ptr<IPartitionSampling> RandomBiPartitionSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix) const {
    uint32 numExamples = labelMatrix.getNumRows();
    uint32 numHoldout = (uint32) (holdoutSetSize_ * numExamples);
    uint32 numTraining = numExamples - numHoldout;
    return std::make_unique<RandomBiPartitionSampling>(numTraining, numHoldout);
}

std::unique_ptr<IPartitionSampling> RandomBiPartitionSamplingFactory::create(const CsrLabelMatrix& labelMatrix) const {
    uint32 numExamples = labelMatrix.getNumRows();
    uint32 numHoldout = (uint32) (holdoutSetSize_ * numExamples);
    uint32 numTraining = numExamples - numHoldout;
    return std::make_unique<RandomBiPartitionSampling>(numTraining, numHoldout);
}
