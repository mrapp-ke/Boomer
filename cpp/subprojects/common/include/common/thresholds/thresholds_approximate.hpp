/*
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/thresholds/thresholds_factory.hpp"
#include "common/binning/feature_binning.hpp"


/**
 * A factory that allows to create instances of the type `ApproximateThresholds`.
 */
class ApproximateThresholdsFactory final : public IThresholdsFactory {

    private:

        std::shared_ptr<IFeatureBinning> binningPtr_;

        uint32 numThreads_;

    public:

        /**
         * @param binningPtr A shared pointer to an object of type `IFeatureBinning` that implements the binning method
         *                   to be used
         * @param numThreads The number of CPU threads to be used to update statistics in parallel. Must be at least 1
         */
        ApproximateThresholdsFactory(std::shared_ptr<IFeatureBinning> binningPtr, uint32 numThreads);

        std::unique_ptr<IThresholds> create(
            std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
            std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
            std::shared_ptr<IStatisticsProvider> statisticsProviderPtr,
            std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr) const override;

};
