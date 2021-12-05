#include "common/sampling/partition_sampling_bi_stratified_example_wise.hpp"
#include "common/sampling/stratified_sampling.hpp"
#include "common/iterator/index_iterator.hpp"
#include "common/validation.hpp"


/**
 * Allows to use stratified sampling, where distinct label vectors are treated as individual classes, to split the
 * training examples into two mutually exclusive sets that may be used as a training set and a holdout set.
 *
 * @tparam LabelMatrix The type of the label matrix that provides random or row-wise access to the labels of the
 *                     training examples
 */
template<typename LabelMatrix>
class ExampleWiseStratifiedBiPartitionSampling final : public IPartitionSampling {

    private:

        BiPartition partition_;

        ExampleWiseStratification<LabelMatrix, IndexIterator> stratification_;

    public:

        /**
         * @param labelMatrix   A reference to an object of template type `LabelMatrix` that provides random or row-wise
         *                      access to the labels of the training examples
         * @param numTraining   The number of examples to be included in the training set
         * @param numHoldout    The number of examples to be included in the holdout set
         */
        ExampleWiseStratifiedBiPartitionSampling(const LabelMatrix& labelMatrix, uint32 numTraining, uint32 numHoldout)
            : partition_(BiPartition(numTraining, numHoldout)),
              stratification_(ExampleWiseStratification<LabelMatrix, IndexIterator>(
                  labelMatrix, IndexIterator(), IndexIterator(labelMatrix.getNumRows()))) {

        }

        IPartition& partition(RNG& rng) override {
            stratification_.sampleBiPartition(partition_, rng);
            return partition_;
        }

};

ExampleWiseStratifiedBiPartitionSamplingFactory::ExampleWiseStratifiedBiPartitionSamplingFactory(float32 holdoutSetSize)
    : holdoutSetSize_(holdoutSetSize) {
    assertGreater<float32>("holdoutSetSize", holdoutSetSize, 0);
    assertLess<float32>("holdoutSetSize", holdoutSetSize, 1);
}

std::unique_ptr<IPartitionSampling> ExampleWiseStratifiedBiPartitionSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix) const {
    uint32 numExamples = labelMatrix.getNumRows();
    uint32 numHoldout = (uint32) (holdoutSetSize_ * numExamples);
    uint32 numTraining = numExamples - numHoldout;
    return std::make_unique<ExampleWiseStratifiedBiPartitionSampling<CContiguousLabelMatrix>>(labelMatrix, numTraining,
                                                                                              numHoldout);
}

std::unique_ptr<IPartitionSampling> ExampleWiseStratifiedBiPartitionSamplingFactory::create(
        const CsrLabelMatrix& labelMatrix) const {
    uint32 numExamples = labelMatrix.getNumRows();
    uint32 numHoldout = (uint32) (holdoutSetSize_ * numExamples);
    uint32 numTraining = numExamples - numHoldout;
    return std::make_unique<ExampleWiseStratifiedBiPartitionSampling<CsrLabelMatrix>>(labelMatrix, numTraining,
                                                                                      numHoldout);
}
