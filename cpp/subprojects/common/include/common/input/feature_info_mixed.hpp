/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_info.hpp"

/**
 * Defines an interface for all classes that provide information about the types of individual features in cases where
 * different types of features, i.e., ordinal, nominal and numerical ones, are available.
 */
class MLRLCOMMON_API IMixedFeatureInfo : public IFeatureInfo {
    public:

        virtual ~IMixedFeatureInfo() override {};

        /**
         * Marks the feature at a specific index as numerical.
         *
         * @param featureIndex The index of the feature
         */
        virtual void setNumerical(uint32 featureIndex) = 0;

        /**
         * Marks the feature at a specific index as ordinal.
         *
         * @param featureIndex The index of the feature
         */
        virtual void setOrdinal(uint32 featureIndex) = 0;

        /**
         * Marks the feature at a specific index as nominal.
         *
         * @param featureIndex The index of the feature
         */
        virtual void setNominal(uint32 featureIndex) = 0;
};

/**
 * Creates and returns a new object of type `IMixedFeatureInfo`.
 *
 * @param numFeatures   The total number of available features
 * @return              An unique pointer to an object of type `IMixedFeatureInfo` that has been created
 */
MLRLCOMMON_API std::unique_ptr<IMixedFeatureInfo> createMixedFeatureInfo(uint32 numFeatures);
