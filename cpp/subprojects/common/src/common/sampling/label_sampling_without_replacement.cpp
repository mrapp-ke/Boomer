#include "common/sampling/label_sampling_without_replacement.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/iterator/index_iterator.hpp"
#include "common/validation.hpp"
#include "index_sampling.hpp"


/**
 * Allows to select a subset of the available labels without replacement.
 */
class LabelSamplingWithoutReplacement final : public ILabelSampling {

    private:

        uint32 numLabels_;

        PartialIndexVector indexVector_;

    public:

        /**
         * @param numLabels     The total number of available labels
         * @param numSamples    The number of labels to be included in the sample
         */
        LabelSamplingWithoutReplacement(uint32 numLabels, uint32 numSamples)
            : numLabels_(numLabels), indexVector_(PartialIndexVector(numSamples)) {

        }

        const IIndexVector& sample(RNG& rng) override {
            sampleIndicesWithoutReplacement<IndexIterator>(indexVector_, IndexIterator(numLabels_), numLabels_, rng);
            return indexVector_;
        }

};

LabelSamplingWithoutReplacementFactory::LabelSamplingWithoutReplacementFactory(uint32 numSamples)
    : numSamples_(numSamples) {
    assertGreaterOrEqual<uint32>("numSamples", numSamples, 1);
}

std::unique_ptr<ILabelSampling> LabelSamplingWithoutReplacementFactory::create(uint32 numLabels) const {
    return std::make_unique<LabelSamplingWithoutReplacement>(numLabels,
                                                             numSamples_ > numLabels ? numLabels : numSamples_);
}
