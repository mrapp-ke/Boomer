/*
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/thresholds/thresholds.hpp"
#include "common/binning/feature_binning.hpp"


/**
 * A factory that allows to create instances of the type `ApproximateThresholds`.
 */
class ApproximateThresholdsFactory final : public IThresholdsFactory {

    private:

        std::unique_ptr<IFeatureBinningFactory> numericalFeatureBinningFactoryPtr_;

        std::unique_ptr<IFeatureBinningFactory> nominalFeatureBinningFactoryPtr_;

        uint32 numThreads_;

    public:

        /**
         * @param numericalFeatureBinningFactoryPtr An unique pointer to an object of type `IFeatureBinningFactory` that
         *                                          allows to create implementations of the binning method to be used
         *                                          for assigning numerical feature values to bins
         * @param nominalFeatureBinningFactoryPtr   An unique pointer to an object of type `IFeatureBinningFactory` that
         *                                          allows to create implementations of the binning method to be used
         *                                          for assigning nominal feature values to bins
         * @param numThreads                        The number of CPU threads to be used to update statistics in
         *                                          parallel. Must be at least 1
         */
        ApproximateThresholdsFactory(std::unique_ptr<IFeatureBinningFactory> numericalFeatureBinningFactoryPtr,
                                     std::unique_ptr<IFeatureBinningFactory> nominalFeatureBinningFactoryPtr,
                                     uint32 numThreads);

        std::unique_ptr<IThresholds> create(const IColumnWiseFeatureMatrix& featureMatrix,
                                            const INominalFeatureMask& nominalFeatureMask,
                                            IStatisticsProvider& statisticsProvider) const override;

};
