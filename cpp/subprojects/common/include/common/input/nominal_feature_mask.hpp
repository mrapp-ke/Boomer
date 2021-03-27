/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * Defines an interface for all classes that provide access to the information whether the features at specific indices
 * are nominal or not.
 */
class INominalFeatureMask {

    public:

        virtual ~INominalFeatureMask() { };

        /**
         * Returns whether the feature at a specific index is nominal or not.
         *
         * @param featureIndex  The index of the feature
         * @return              True, if the feature at the given index is nominal, false otherwise
         */
        virtual bool isNominal(uint32 featureIndex) const = 0;

};
