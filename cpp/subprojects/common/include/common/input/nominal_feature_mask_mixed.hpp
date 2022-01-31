/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/nominal_feature_mask.hpp"
#include <memory>


/**
 * Defines an interface for all classes that allow to check whether individual features are nominal or not in cases
 * where different types of features, i.e., nominal and numerical/ordinal ones, are available.
 */
class MLRLCOMMON_API IMixedNominalFeatureMask : public INominalFeatureMask {

    public:

        virtual ~IMixedNominalFeatureMask() override { };

        /**
         * Sets whether the feature at a specific index is nominal or not.
         *
         * @param featureIndex  The index of the feature
         * @param nominal       True, if the feature is nominal, false, if it is numerical/ordinal
         */
        virtual void setNominal(uint32 featureIndex, bool nominal) = 0;

};

/**
 * Creates and returns a new object of type `IMixedNominalFeatureMask`.
 *
 * @param numFeatures   The total number of available features
 * @return              An unique pointer to an object of type `IMixedNominalFeatureMask` that has been created
 */
MLRLCOMMON_API std::unique_ptr<IMixedNominalFeatureMask> createMixedNominalFeatureMask(uint32 numFeatures);
