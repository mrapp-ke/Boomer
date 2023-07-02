#include "common/sampling/label_sampling_round_robin.hpp"

#include "common/indices/index_vector_partial.hpp"

/**
 * Allows to select a sinle label in a round-robin fashion.
 */
class RoundRobinLabelSampling final : public ILabelSampling {
    private:

        const uint32 numLabels_;

        PartialIndexVector indexVector_;

        uint32 nextIndex_;

    public:

        /**
         * @param numLabels The total number of available labels
         */
        RoundRobinLabelSampling(uint32 numLabels)
            : numLabels_(numLabels), indexVector_(PartialIndexVector(1)), nextIndex_(0) {}

        const IIndexVector& sample(RNG& rng) override {
            indexVector_.begin()[0] = nextIndex_;
            nextIndex_++;

            if (nextIndex_ >= numLabels_) {
                nextIndex_ = 0;
            }

            return indexVector_;
        }
};

/**
 * Allows to create objects of type `ILabelSampling` that select a single label in a round-robin fashion.
 */
class RoundRobinLabelSamplingFactory final : public ILabelSamplingFactory {
    private:

        const uint32 numLabels_;

    public:

        /**
         * @param numLabels The total number of available labels
         */
        RoundRobinLabelSamplingFactory(uint32 numLabels) : numLabels_(numLabels) {}

        std::unique_ptr<ILabelSampling> create() const override {
            return std::make_unique<RoundRobinLabelSampling>(numLabels_);
        }
};

std::unique_ptr<ILabelSamplingFactory> RoundRobinLabelSamplingConfig::createLabelSamplingFactory(
  const ILabelMatrix& labelMatrix) const {
    return std::make_unique<RoundRobinLabelSamplingFactory>(labelMatrix.getNumCols());
}
