#include "common/sampling/feature_sampling_no.hpp"
#include "common/indices/index_vector_full.hpp"


std::unique_ptr<IIndexVector> NoFeatureSubSampling::subSample(uint32 numFeatures, RNG& rng) const {
    return std::make_unique<FullIndexVector>(numFeatures);
}
