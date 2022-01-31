/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_two_dimensional.hpp"


/**
 * Defines an interface for all label matrices.
 */
class MLRLCOMMON_API ILabelMatrix : virtual public ITwoDimensionalView {

    public:

        virtual ~ILabelMatrix() override { };

        /**
         * Returns whether the label matrix is sparse or not.
         *
         * @return True, if the label matrix is sparse, false otherwise
         */
        virtual bool isSparse() const = 0;

};
