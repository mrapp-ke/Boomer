/*
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/thresholds/thresholds_factory.hpp"
#include "common/binning/feature_binning.hpp"


/**
 * A factory that allows to create instances of the type `ApproximateThresholds`.
 */
class ApproximateThresholdsFactory final : public IThresholdsFactory {

    private:

        std::unique_ptr<IFeatureBinning> binningPtr_;

        uint32 numThreads_;

    public:

        /**
         * @param binningPtr An unique pointer to an object of type `IFeatureBinning` that implements the binning method
         *                   to be used
         * @param numThreads The number of CPU threads to be used to update statistics in parallel. Must be at least 1
         */
        ApproximateThresholdsFactory(std::unique_ptr<IFeatureBinning> binningPtr, uint32 numThreads);

        std::unique_ptr<IThresholds> create(const IFeatureMatrix& featureMatrix,
                                            const INominalFeatureMask& nominalFeatureMask,
                                            IStatisticsProvider& statisticsProvider) const override;

};
