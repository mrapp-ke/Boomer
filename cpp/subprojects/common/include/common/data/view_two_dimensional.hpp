/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include "common/macros.hpp"

/**
 * Defines an interface for all two-dimensional views.
 */
class MLRLCOMMON_API ITwoDimensionalView {
    public:

        virtual ~ITwoDimensionalView() {};

        /**
         * Returns the number of rows in the view.
         *
         * @return The number of rows
         */
        virtual uint32 getNumRows() const = 0;

        /**
         * Returns the number of columns in the view.
         *
         * @return The number of columns
         */
        virtual uint32 getNumCols() const = 0;
};
