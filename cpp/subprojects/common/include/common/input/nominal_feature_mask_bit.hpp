/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/vector_bit.hpp"
#include "common/input/nominal_feature_mask.hpp"


/**
 * Provides access to the information whether the features at specific indices are nominal or not, based on a
 * `BitVector` that stores whether individual features are nominal or not.
 */
class BitNominalFeatureMask : public INominalFeatureMask {

    private:

        BitVector vector_;

    public:

        /**
         * @param numFeatures The total number of available features
         */
        BitNominalFeatureMask(uint32 numFeatures);

        /**
         * Marks the feature at a specific index as nominal.
         *
         * @param featureIndex The index of the feature
         */
        void setNominal(uint32 featureIndex);

        bool isNominal(uint32 featureIndex) const override;

};
