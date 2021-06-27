/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"
#include "common/input/label_matrix.hpp"


/**
 * Implements random read-only access to the labels of individual training examples that are stored in a pre-allocated
 * C-contiguous array.
 */
class CContiguousLabelMatrix final : public IRandomAccessLabelMatrix {

    private:

        CContiguousView<uint8> view_;

    public:

        /**
         * @param numRows   The number of rows in the label matrix
         * @param numCols   The number of columns in the label matrix
         * @param array     A pointer to a C-contiguous array of type `uint8` that stores the labels
         */
        CContiguousLabelMatrix(uint32 numRows, uint32 numCols, uint8* array);

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        uint8 getValue(uint32 exampleIndex, uint32 labelIndex) const override;

};
