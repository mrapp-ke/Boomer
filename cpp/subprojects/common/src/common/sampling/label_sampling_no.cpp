#include "common/sampling/label_sampling_no.hpp"
#include "common/indices/index_vector_complete.hpp"


/**
 * An implementation of the class `ILabelSampling` that does not perform any sampling, but includes all labels.
 */
class NoLabelSampling final : public ILabelSampling {

    private:

        CompleteIndexVector indexVector_;

    public:

        /**
         * @param numLabels The total number of available labels
         */
        NoLabelSampling(uint32 numLabels)
            : indexVector_(numLabels) {

        }

        const IIndexVector& sample(RNG& rng) override {
            return indexVector_;
        }

};

std::unique_ptr<ILabelSampling> NoLabelSamplingFactory::create(uint32 numLabels) const {
    return std::make_unique<NoLabelSampling>(numLabels);
}
