/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * Implements column-wise read-only access to the values that are stored in a pre-allocated matrix in the compressed
 * sparse column (CSC) format.
 *
 * @tparam T The type of the values
 */
template<class T>
class CscView final {

    private:

        uint32 numRows_;

        uint32 numCols_;

        const T* data_;

        const uint32* rowIndices_;

        const uint32* colIndices_;

    public:

        /**
         * @param numRows       The number of rows in the view
         * @param numCols       The number of cols in the view
         * @param data          A pointer to an array of template type `T`, shape `(num_non_zero_values)`, that stores
         *                      all non-zero values
         * @param rowIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
         *                      row-indices, the values in `data` correspond to
         * @param colIndices    A pointer to an array of type `uint32`, shape `(numCols + 1)`, that stores the indices
         *                      of the first element in `data` and `rowIndices` that corresponds to a certain column.
         *                      The index at the last position is equal to `num_non_zero_values`
         */
        CscView(uint32 numRows, uint32 numCols, const T* data, const uint32* rowIndices, const uint32* colIndices);

        /**
         * An iterator that provides read-only access to the values in the view.
         */
        typedef const float32* value_const_iterator;

        /**
         * An iterator that provides read-only access to the indices in the view.
         */
        typedef const uint32* index_const_iterator;

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
         * Returns the number of rows in the view.
         *
         * @return The number of rows
         */
        uint32 getNumRows() const;

        /**
         * Returns the number of columns in the view.
         *
         * @return The number of columns
         */
        uint32 getNumCols() const;

        /**
         * Returns the number of elements at a specific column that correspond to a non-zero value.
         *
         * @param col   The column
         * @return      The number of non-zero elements
         */
        uint32 getNumNonZeroElements(uint32 col) const;

};
