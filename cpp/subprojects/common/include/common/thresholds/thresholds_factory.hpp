/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/thresholds/thresholds.hpp"
#include "common/input/feature_matrix.hpp"
#include "common/input/nominal_feature_mask.hpp"
#include "common/head_refinement/head_refinement_factory.hpp"
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
         * @param featureMatrixPtr          A shared pointer to an object of type `IFeatureMatrix` that provides access
         *                                  to the feature values of the training examples
         * @param nominalFeatureMaskPtr     A shared pointer to an object of type `INominalFeatureMask` that provides
         *                                  access to the information whether individual features are nominal or not
         * @param statisticsProviderPtr     A shared pointer to an object of type `IStatisticsProvider` that provides
         *                                  access to statistics about the labels of the training examples
         * @param headRefinementFactoryPtr  A shared pointer to an object of type `IHeadRefinementFactory` that allows
         *                                  to create instances of the class that should be used to find the heads of
         *                                  rules
         * @return                          An unique pointer to an object of type `IThresholds` that has been created
         */
        virtual std::unique_ptr<IThresholds> create(
            std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
            std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
            std::shared_ptr<IStatisticsProvider> statisticsProviderPtr,
            std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr) const = 0;

};
