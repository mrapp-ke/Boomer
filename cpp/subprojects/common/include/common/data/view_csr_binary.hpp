/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_two_dimensional.hpp"

/**
 * Implements row-wise read-only access to binary values that are stored in a pre-allocated matrix in the compressed
 * sparse row (CSR) format.
 */
class MLRLCOMMON_API BinaryCsrConstView : virtual public ITwoDimensionalView {
    protected:

        /**
         * The number of rows in the view.
         */
        const uint32 numRows_;

        /**
         * The number of columns in the view.
         */
        const uint32 numCols_;

        /**
         * A pointer to an array that stores the indices of the first element in `colIndices_` that corresponds to a
         * certain row.
         */
        uint32* rowIndices_;

        /**
         * A pointer to an array that stores the column-indices, the non-zero elements correspond to.
         */
        uint32* colIndices_;

    public:

        /**
         * @param numRows       The number of rows in the view
         * @param numCols       The number of columns in the view
         * @param rowIndices    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices
         *                      of the first element in `colIndices` that corresponds to a certain row. The index at the
         *                      last position is equal to `num_non_zero_values`
         * @param colIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
         *                      column-indices, the non-zero elements correspond to
         */
        BinaryCsrConstView(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices);

        virtual ~BinaryCsrConstView() override {};

        /**
         * An iterator that provides read-only access to the indices in the view.
         */
        typedef const uint32* index_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices at a specific row.
         *
         * @param row   The row
         * @return      An `index_const_iterator` to the beginning of the indices
         */
        index_const_iterator indices_cbegin(uint32 row) const;

        /**
         * Returns an `index_const_iterator` to the end of the indices at a specific row.
         *
         * @param row   The row
         * @return      An `index_const_iterator` to the end of the indices
         */
        index_const_iterator indices_cend(uint32 row) const;

        /**
         * Returns the number of non-zero elements in the view.
         *
         * @return The number of non-zero elements
         */
        uint32 getNumNonZeroElements() const;

        uint32 getNumRows() const override final;

        uint32 getNumCols() const override final;
};

/**
 * Implements row-wise read and write access to binary values that are stored in a pre-allocated matrix in the
 * compressed sparse row (CSR) format.
 */
class BinaryCsrView : public BinaryCsrConstView {
    public:

        /**
         * @param numRows       The number of rows in the view
         * @param numCols       The number of columns in the view
         * @param rowIndices    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices
         *                      of the first element in `colIndices` that corresponds to a certain row. The index at the
         *                      last position is equal to `num_non_zero_values`
         * @param colIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
         *                      column-indices, the non-zero elements correspond to
         */
        BinaryCsrView(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices);

        virtual ~BinaryCsrView() override {};

        /**
         * An iterator that provides access to the indices of the view and allows to modify them.
         */
        typedef uint32* index_iterator;

        /**
         * Returns an `index_iterator` to the beginning of the indices at a specific row.
         *
         * @param row   The row
         * @return      An `index_iterator` to the beginning of the indices
         */
        index_iterator indices_begin(uint32 row);

        /**
         * Returns an `index_iterator` to the end of the indices at a specific row.
         *
         * @param row   The row
         * @return      An `index_iterator` to the end of the indices
         */
        index_iterator indices_end(uint32 row);
};
