#include "common/sampling/partition_sampling_bi_stratified_label_wise.hpp"

#include "common/iterator/index_iterator.hpp"
#include "common/sampling/stratified_sampling_label_wise.hpp"
#include "common/util/validation.hpp"

/**
 * Allows to use stratified sampling to split the training examples into two mutually exclusive sets that may be used as
 * a training set and a holdout set, such that for each label the proportion of relevant and irrelevant examples is
 * maintained.
 *
 * @tparam LabelMatrix The type of the label matrix that provides random or row-wise access to the labels of the
 *                     training examples
 */
template<typename LabelMatrix>
class LabelWiseStratifiedBiPartitionSampling final : public IPartitionSampling {
    private:

        BiPartition partition_;

        const LabelWiseStratification<LabelMatrix, IndexIterator> stratification_;

    public:

        /**
         * @param labelMatrix   A reference to an object of template type `LabelMatrix` that provides random or row-wise
         *                      access to the labels of the training examples
         * @param numTraining   The number of examples to be included in the training set
         * @param numHoldout    The number of examples to be included in the holdout set
         */
        LabelWiseStratifiedBiPartitionSampling(const LabelMatrix& labelMatrix, uint32 numTraining, uint32 numHoldout)
            : partition_(BiPartition(numTraining, numHoldout)),
              stratification_(LabelWiseStratification<LabelMatrix, IndexIterator>(
                labelMatrix, IndexIterator(), IndexIterator(labelMatrix.getNumRows()))) {}

        IPartition& partition(RNG& rng) override {
            stratification_.sampleBiPartition(partition_, rng);
            return partition_;
        }
};

/**
 * Allows to create objects of the type `IPartitionSampling` that use stratified sampling to split the training examples
 * into two mutually exclusive sets that may be used as a training set and a holdout set, such that for each label the
 * proportion of relevant and irrelevant examples is maintained.
 */
class LabelWiseStratifiedBiPartitionSamplingFactory final : public IPartitionSamplingFactory {
    private:

        const float32 holdoutSetSize_;

    public:

        /**
         * @param holdoutSetSize The fraction of examples to be included in the holdout set (e.g. a value of 0.6
         *                       corresponds to 60 % of the available examples). Must be in (0, 1)
         */
        LabelWiseStratifiedBiPartitionSamplingFactory(float32 holdoutSetSize) : holdoutSetSize_(holdoutSetSize) {}

        std::unique_ptr<IPartitionSampling> create(const CContiguousLabelMatrix& labelMatrix) const override {
            uint32 numExamples = labelMatrix.getNumRows();
            uint32 numHoldout = (uint32) (holdoutSetSize_ * numExamples);
            uint32 numTraining = numExamples - numHoldout;
            return std::make_unique<LabelWiseStratifiedBiPartitionSampling<CContiguousLabelMatrix>>(
              labelMatrix, numTraining, numHoldout);
        }

        std::unique_ptr<IPartitionSampling> create(const CsrLabelMatrix& labelMatrix) const override {
            uint32 numExamples = labelMatrix.getNumRows();
            uint32 numHoldout = (uint32) (holdoutSetSize_ * numExamples);
            uint32 numTraining = numExamples - numHoldout;
            return std::make_unique<LabelWiseStratifiedBiPartitionSampling<CsrLabelMatrix>>(labelMatrix, numTraining,
                                                                                            numHoldout);
        }
};

LabelWiseStratifiedBiPartitionSamplingConfig::LabelWiseStratifiedBiPartitionSamplingConfig() : holdoutSetSize_(0.33f) {}

float32 LabelWiseStratifiedBiPartitionSamplingConfig::getHoldoutSetSize() const {
    return holdoutSetSize_;
}

ILabelWiseStratifiedBiPartitionSamplingConfig& LabelWiseStratifiedBiPartitionSamplingConfig::setHoldoutSetSize(
  float32 holdoutSetSize) {
    assertGreater<float32>("holdoutSetSize", holdoutSetSize, 0);
    assertLess<float32>("holdoutSetSize", holdoutSetSize, 1);
    holdoutSetSize_ = holdoutSetSize;
    return *this;
}

std::unique_ptr<IPartitionSamplingFactory>
  LabelWiseStratifiedBiPartitionSamplingConfig::createPartitionSamplingFactory() const {
    return std::make_unique<LabelWiseStratifiedBiPartitionSamplingFactory>(holdoutSetSize_);
}
