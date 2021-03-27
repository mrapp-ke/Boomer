/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/matrix_dok_binary.hpp"
#include "common/input/label_matrix.hpp"


/**
 * Implements random access to the labels of individual training examples that are stored in a pre-allocated sparse
 * matrix in the dictionary of keys (DOK) format.
 */
class DokLabelMatrix : public IRandomAccessLabelMatrix {

    private:

        uint32 numRows_;

        uint32 numCols_;

        BinaryDokMatrix matrix_;

    public:

        /**
         * @param numRows   The number of rows in the label matrix
         * @param numCols   The number of columns in the label matrix
         */
        DokLabelMatrix(uint32 numRows, uint32 numCols);

        /**
         * Marks a label of an example as relevant.
         *
         * @param exampleIndex  The index of the example
         * @param labelIndex    The index of the label
         */
        void setValue(uint32 exampleIndex, uint32 labelIndex);

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        uint8 getValue(uint32 exampleIndex, uint32 labelIndex) const override;

};
