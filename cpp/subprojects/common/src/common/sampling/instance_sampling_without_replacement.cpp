#include "common/sampling/instance_sampling_without_replacement.hpp"

#include "common/iterator/index_iterator.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/sampling/weight_sampling.hpp"
#include "common/util/validation.hpp"

static inline void sampleInternally(const SinglePartition& partition, float32 sampleSize, BitWeightVector& weightVector,
                                    RNG& rng) {
    uint32 numExamples = partition.getNumElements();
    uint32 numSamples = (uint32) (sampleSize * numExamples);
    sampleWeightsWithoutReplacement<SinglePartition::const_iterator>(weightVector, partition.cbegin(), numExamples,
                                                                     numSamples, rng);
}

static inline void sampleInternally(BiPartition& partition, float32 sampleSize, BitWeightVector& weightVector,
                                    RNG& rng) {
    uint32 numTrainingExamples = partition.getNumFirst();
    uint32 numSamples = (uint32) (sampleSize * numTrainingExamples);
    sampleWeightsWithoutReplacement<BiPartition::const_iterator>(weightVector, partition.first_cbegin(),
                                                                 numTrainingExamples, numSamples, rng);
}

/**
 * Allows to select a subset of the available training examples without replacement.
 *
 * @tparam Partition The type of the object that provides access to the indices of the examples that are included in the
 *                   training set
 */
template<typename Partition>
class InstanceSamplingWithoutReplacement final : public IInstanceSampling {
    private:

        Partition& partition_;

        const float32 sampleSize_;

        BitWeightVector weightVector_;

    public:

        /**
         * @param partition  A reference to an object of template type `Partition` that provides access to the indices
         *                   of the examples that are included in the training set
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1)
         */
        InstanceSamplingWithoutReplacement(Partition& partition, float32 sampleSize)
            : partition_(partition), sampleSize_(sampleSize),
              weightVector_(BitWeightVector(partition.getNumElements())) {}

        const IWeightVector& sample(RNG& rng) override {
            sampleInternally(partition_, sampleSize_, weightVector_, rng);
            return weightVector_;
        }
};

/**
 * Allows to create instances of the type `IInstanceSampling` that allow to select a subset of the available training
 * examples without replacement.
 */
class InstanceSamplingWithoutReplacementFactory final : public IInstanceSamplingFactory {
    private:

        const float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1)
         */
        InstanceSamplingWithoutReplacementFactory(float32 sampleSize) : sampleSize_(sampleSize) {}

        std::unique_ptr<IInstanceSampling> create(const CContiguousLabelMatrix& labelMatrix,
                                                  const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<InstanceSamplingWithoutReplacement<const SinglePartition>>(partition, sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousLabelMatrix& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<InstanceSamplingWithoutReplacement<BiPartition>>(partition, sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrLabelMatrix& labelMatrix, const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<InstanceSamplingWithoutReplacement<const SinglePartition>>(partition, sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrLabelMatrix& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<InstanceSamplingWithoutReplacement<BiPartition>>(partition, sampleSize_);
        }
};

InstanceSamplingWithoutReplacementConfig::InstanceSamplingWithoutReplacementConfig() : sampleSize_(0.66f) {}

float32 InstanceSamplingWithoutReplacementConfig::getSampleSize() const {
    return sampleSize_;
}

IInstanceSamplingWithoutReplacementConfig& InstanceSamplingWithoutReplacementConfig::setSampleSize(float32 sampleSize) {
    assertGreater<float32>("sampleSize", sampleSize, 0);
    assertLess<float32>("sampleSize", sampleSize, 1);
    sampleSize_ = sampleSize;
    return *this;
}

std::unique_ptr<IInstanceSamplingFactory> InstanceSamplingWithoutReplacementConfig::createInstanceSamplingFactory()
  const {
    return std::make_unique<InstanceSamplingWithoutReplacementFactory>(sampleSize_);
}
