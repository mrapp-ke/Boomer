/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_csc_binary.hpp"
#include "common/input/label_matrix_c_contiguous.hpp"
#include "common/input/label_matrix_csr.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"


/**
 * Implements column-wise read-only access to the labels of individual training examples that are stored in a matrix in
 * the compressed sparse column (CSC) format.
 *
 * This class provides copy constructors for copying an existing `CContiguousLabelMatrix`, which provides random access,
 * or a `CsrLabelMatrix`, which provides row-wise access to the labels of the training examples. These constructors
 * expect the indices of the examples to be considered when copying the existing label matrix to be provided.
 */
class CscLabelMatrix final {

    private:

        uint32* rowIndices_;

        uint32* colIndices_;

        BinaryCscConstView view_;

    public:

        /**
         * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` to be copied
         * @param indicesBegin  A `CompleteIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `CompleteIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const CContiguousLabelMatrix& labelMatrix, CompleteIndexVector::const_iterator indicesBegin,
                       CompleteIndexVector::const_iterator indicesEnd);

        /**
         * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` to be copied
         * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const CContiguousLabelMatrix& labelMatrix, PartialIndexVector::const_iterator indicesBegin,
                       PartialIndexVector::const_iterator indicesEnd);

        /**
         * @param labelMatrix   A reference to an object of type `CsrLabelMatrix` to be copied
         * @param indicesBegin  A `CompleteIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `CompleteIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const CsrLabelMatrix& labelMatrix, CompleteIndexVector::const_iterator indicesBegin,
                       CompleteIndexVector::const_iterator indicesEnd);

        /**
         * @param labelMatrix   A reference to an object of type `CsrLabelMatrix` to be copied
         * @param indicesBegin  A `PartialIndexVector::const_iterator` to the beginning of the indices of the examples
         *                      to be considered
         * @param indicesEnd    A `PartialIndexVector::const_iterator` to the end of the indices of the examples to be
         *                      considered
         */
        CscLabelMatrix(const CsrLabelMatrix& labelMatrix, PartialIndexVector::const_iterator indicesBegin,
                       PartialIndexVector::const_iterator indicesEnd);

        ~CscLabelMatrix();

        /**
         * An iterator that provides read-only access to the indices of the relevant labels.
         */
        typedef BinaryCscConstView::index_const_iterator index_const_iterator;

        /**
         * An iterator that provides read-only access to the values in the label matrix.
         */
        typedef BinaryCscConstView::value_const_iterator value_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices at a specific column.
         *
         * @param col   The column
         * @return      An `index_const_iterator` to the beginning of the indices
         */
        index_const_iterator column_indices_cbegin(uint32 col) const;

        /**
         * Returns an `index_const_iterator` to the end of the indices at a specific column.
         *
         * @param col   The column
         * @return      An `index_const_iterator` to the end of the indices
         */
        index_const_iterator column_indices_cend(uint32 col) const;

        /**
         * Returns a `value_const_iterator` to the beginning of the values at a specific column.
         *
         * @param col   The column
         * @return      A `value_const_iterator` to the beginning of the values
         */
        value_const_iterator column_values_cbegin(uint32 col) const;

        /**
         * Returns a `value_const_iterator` to the end of the values at a specific column.
         *
         * @param col   The column
         * @return      A `value_const_iterator` to the end of the values
         */
        value_const_iterator column_values_cend(uint32 col) const;

        /**
         * Returns the number of rows in the label matrix.
         *
         * @return The number of rows
         */
        uint32 getNumRows() const;

        /**
         * Returns the number of columns in the label matrix.
         *
         * @return The number of columns
         */
        uint32 getNumCols() const;

        /**
         * Returns the number of relevant labels.
         *
         * @return The number of relevant labels
         */
        uint32 getNumNonZeroElements() const;

};
