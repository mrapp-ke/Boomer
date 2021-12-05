/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/input/nominal_feature_mask.hpp"


/**
 * Provides access to the information whether the features at specific indices are nominal or not, if all features are
 * either nominal or if all features are not nominal.
 */
class EqualNominalFeatureMask : public INominalFeatureMask {

    private:

        bool nominal_;

    public:

        /**
         * @param nominal True, if all features are nominal, false, if all features are not nominal
         */
        EqualNominalFeatureMask(bool nominal);

        bool isNominal(uint32 featureIndex) const override;

};
