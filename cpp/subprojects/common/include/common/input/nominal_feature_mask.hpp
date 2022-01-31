/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include "common/macros.hpp"


/**
 * Defines an interface for all classes that allow to check whether individual features are nominal or not.
 */
class MLRLCOMMON_API INominalFeatureMask {

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
