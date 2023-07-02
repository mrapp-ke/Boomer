#include "common/sampling/label_sampling_no.hpp"

#include "common/indices/index_vector_complete.hpp"

/**
 * An implementation of the class `ILabelSampling` that does not perform any sampling, but includes all labels.
 */
class NoLabelSampling final : public ILabelSampling {
    private:

        const CompleteIndexVector indexVector_;

    public:

        /**
         * @param numLabels The total number of available labels
         */
        NoLabelSampling(uint32 numLabels) : indexVector_(numLabels) {}

        const IIndexVector& sample(RNG& rng) override {
            return indexVector_;
        }
};

/**
 * Allows to create objects of the class `ILabelSampling` that do not perform any sampling, but include all labels.
 */
class NoLabelSamplingFactory final : public ILabelSamplingFactory {
    private:

        const uint32 numLabels_;

    public:

        /**
         * @param numLabels The total number of available labels
         */
        NoLabelSamplingFactory(uint32 numLabels) : numLabels_(numLabels) {}

        std::unique_ptr<ILabelSampling> create() const override {
            return std::make_unique<NoLabelSampling>(numLabels_);
        }
};

std::unique_ptr<ILabelSamplingFactory> NoLabelSamplingConfig::createLabelSamplingFactory(
  const ILabelMatrix& labelMatrix) const {
    return std::make_unique<NoLabelSamplingFactory>(labelMatrix.getNumCols());
}
