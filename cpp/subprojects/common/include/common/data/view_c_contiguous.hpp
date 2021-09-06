/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * Implements row-wise read-only access to the values that are stored in a pre-allocated C-contiguous array.
 *
 * @tparam T The type of the values
 */
template<typename T>
class CContiguousConstView {

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
         * A pointer to the array that stores the values, the view provides access to.
         */
        T* array_;

    public:

        /**
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         * @param array     A pointer to a C-contiguous array of template type `T` that stores the values, the view
         *                  provides access to
         */
        CContiguousConstView(uint32 numRows, uint32 numCols, T* array);

        /**
         * An iterator that provides read-only access to the elements in the view.
         */
        typedef const T* const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the beginning of the given row
         */
        const_iterator row_cbegin(uint32 row) const;

        /**
         * Returns a `const_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the end of the given row
         */
        const_iterator row_cend(uint32 row) const;

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

};

/**
 * Implements row-wise read and write access to the values that are stored in a pre-allocated C-contiguous array.
 *
 * @tparam T The type of the values
 */
template<typename T>
class CContiguousView : public CContiguousConstView<T> {

    public:

        /**
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         * @param array     A pointer to a C-contiguous array of template type `T` that stores the values, the view
         *                  provides access to
         */
        CContiguousView(uint32 numRows, uint32 numCols, T* array);

        /**
         * An iterator that provides access to the elements in the view and allows to modify them.
         */
        typedef T* iterator;

        /**
         * Returns an `iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the beginning of the given row
         */
        iterator row_begin(uint32 row);

        /**
         * Returns an `iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the end of the given row
         */
        iterator row_end(uint32 row);

};
