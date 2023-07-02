#include "common/sampling/feature_sampling_predefined.hpp"

PredefinedFeatureSampling::PredefinedFeatureSampling(const IIndexVector& indexVector) : indexVector_(indexVector) {}

const IIndexVector& PredefinedFeatureSampling::sample(RNG& rng) {
    return indexVector_;
}

std::unique_ptr<IFeatureSampling> PredefinedFeatureSampling::createBeamSearchFeatureSampling(RNG& rng, bool resample) {
    return std::make_unique<PredefinedFeatureSampling>(indexVector_);
}
