/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/vector_dok_binary.hpp"
#include "common/input/nominal_feature_mask.hpp"


/**
 * Provides access to the information whether the features at specific indices are nominal or not, based on a
 * `BinaryDokVector` that stores the indices of all nominal features.
 */
class DokNominalFeatureMask : public INominalFeatureMask {

    private:

        BinaryDokVector vector_;

    public:

        /**
         * Marks the feature at a specific index as nominal.
         *
         * @param featureIndex The index of the feature
         */
        void setNominal(uint32 featureIndex);

        bool isNominal(uint32 featureIndex) const override;

};
