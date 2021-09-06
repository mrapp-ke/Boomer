/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/thresholds/thresholds.hpp"
#include "common/input/feature_matrix.hpp"
#include "common/input/nominal_feature_mask.hpp"
#include "common/statistics/statistics_provider.hpp"


/**
 * Defines an interface for all classes that allow to create instances of the type `IThresholds`.
 */
class IThresholdsFactory {

    public:

        virtual ~IThresholdsFactory() { };

        /**
         * Creates and returns a new object of type `IThresholds`.
         *
         * @param featureMatrix         A reference to an object of type `IFeatureMatrix` that provides access to the
         *                              feature values of the training examples
         * @param nominalFeatureMask    A reference  to an object of type `INominalFeatureMask` that provides access to
         *                              the information whether individual features are nominal or not
         * @param statisticsProvider    A reference to an object of type `IStatisticsProvider` that provides access to
         *                              statistics about the labels of the training examples
         * @return                      An unique pointer to an object of type `IThresholds` that has been created
         */
        virtual std::unique_ptr<IThresholds> create(const IFeatureMatrix& featureMatrix,
                                                    const INominalFeatureMask& nominalFeatureMask,
                                                    IStatisticsProvider& statisticsProvider) const = 0;

};
