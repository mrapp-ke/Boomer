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
template<typename T>
class CscConstView {

    protected:

        /**
         * The number of rows in the view.
         */
        uint32 numRows_;

        /**
         * The number of columns in the view.
         */
        uint32 numCols_;

        /**
         * A pointer to an array that stores all non-zero values.
         */
        T* data_;

        /**
         * A pointer to an array that stores the row-indices, the values in `data_` correspond to.
         */
        uint32* rowIndices_;

        /**
         * A pointer to an array that stores the indices of the first element in `data_` and `rowIndices_` that
         * corresponds to a certain column.
         */
        uint32* colIndices_;

    public:

        /**
         * @param numRows       The number of rows in the view
         * @param numCols       The number of columns in the view
         * @param data          A pointer to an array of template type `T`, shape `(num_non_zero_values)`, that stores
         *                      all non-zero values
         * @param rowIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
         *                      row-indices, the values in `data` correspond to
         * @param colIndices    A pointer to an array of type `uint32`, shape `(numCols + 1)`, that stores the indices
         *                      of the first element in `data` and `rowIndices` that corresponds to a certain column.
         *                      The index at the last position is equal to `num_non_zero_values`
         */
        CscConstView(uint32 numRows, uint32 numCols, T* data, uint32* rowIndices, uint32* colIndices);

        /**
         * An iterator that provides read-only access to the values in the view.
         */
        typedef const T* value_const_iterator;

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
         * Returns the number of non-zero elements in the view.
         *
         * @return The number of non-zero elements
         */
        uint32 getNumNonZeroElements() const;

};


/**
 * Implements column-wise read and write access to the values that are stored in a pre-allocated matrix in the
 * compressed sparse column (CSC) format.
 *
 * @tparam T The type of the values
 */
template<typename T>
class CscView final : public CscConstView<T> {

    public:

        /**
         * @param numRows       The number of rows in the view
         * @param numCols       The number of columns in the view
         * @param data          A pointer to an array of template type `T`, shape `(num_non_zero_values)`, that stores
         *                      all non-zero values
         * @param rowIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
         *                      row-indices, the values in `data` correspond to
         * @param colIndices    A pointer to an array of type `uint32`, shape `(numCols + 1)`, that stores the indices
         *                      of the first element in `data` and `rowIndices` that corresponds to a certain column.
         *                      The index at the last position is equal to `num_non_zero_values`
         */
        CscView(uint32 numRows, uint32 numCols, T* data, uint32* rowIndices, uint32* colIndices);

        /**
         * An iterator that provides access to the values in the view and allows to modify them.
         */
        typedef T* value_iterator;

        /**
         * An iterator that provides access to the indices in the view and allows to modify them.
         */
        typedef uint32* index_iterator;

        /**
         * Returns a `value_iterator` to the beginning of the values at a specific column.
         *
         * @param col   The column
         * @return      A `value_iterator` to the beginning of the values
         */
        value_iterator column_values_begin(uint32 col);

        /**
         * Returns a `value_iterator` to the end of the values at a specific column.
         *
         * @param col   The column
         * @return      A `value_iterator` to the end of the values
         */
        value_iterator column_values_end(uint32 col);

        /**
         * Returns an `index_iterator` to the beginning of the indices at a specific column.
         *
         * @param col   The column
         * @return      An `index_iterator` to the beginning of the indices
         */
        index_iterator column_indices_begin(uint32 col);

        /**
         * Returns an `index_iterator` to the end of the indices at a specific column.
         *
         * @param col   The column
         * @return      An `index_iterator` to the end of the indices
         */
        index_iterator column_indices_end(uint32 col);

};
