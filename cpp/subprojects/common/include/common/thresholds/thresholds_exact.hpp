/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/thresholds/thresholds_factory.hpp"


/**
 * A factory that allows to create instances of the type `ExactThresholds`.
 */
class ExactThresholdsFactory final : public IThresholdsFactory {

    private:

        uint32 numThreads_;

    public:

        /**
         * @param numThreads The number of CPU threads to be used to update statistics in parallel. Must be at least 1
         */
        ExactThresholdsFactory(uint32 numThreads);

        std::unique_ptr<IThresholds> create(const IFeatureMatrix& featureMatrix,
                                            const INominalFeatureMask& nominalFeatureMask,
                                            IStatisticsProvider& statisticsProvider) const override;

};
