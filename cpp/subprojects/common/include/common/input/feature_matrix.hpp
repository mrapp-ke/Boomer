/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_two_dimensional.hpp"


/**
 * Defines an interface for all feature matrices.
 */
class MLRLCOMMON_API IFeatureMatrix : virtual public ITwoDimensionalView {

    public:

        virtual ~IFeatureMatrix() override { };

        /**
         * Returns whether the feature matrix is sparse or not.
         *
         * @return True, if the feature matrix is sparse, false otherwise
         */
        virtual bool isSparse() const = 0;

};
