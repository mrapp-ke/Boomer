/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * Implements column-wise read-only access to the values that are stored in a pre-allocated Fortran-contiguous array.
 *
 * @tparam T The type of the values
 */
template<typename T>
class FortranContiguousConstView {

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
         * A pointer to an array that stores the values.
         */
        T* array_;

    public:

        /**
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         * @param array     A pointer to a Fortran-contiguous array of template type `T` that stores the values, the
         *                  view provides access to
         */
        FortranContiguousConstView(uint32 numRows, uint32 numCols, T* array);

        /**
         * An iterator that provides read-only access to the values in the view.
         */
        typedef const T* const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of a specific column.
         *
         * @param col   The column
         * @return      A `const_iterator` to the beginning
         */
        const_iterator column_cbegin(uint32 col) const;

        /**
         * Returns a `const_iterator` to the end of a specific column.
         *
         * @param col   The column
         * @return      A `const_iterator` to the end
         */
        const_iterator column_cend(uint32 col) const;

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
 * Implements column-wise read and write access to the values that are stored in a pre-allocated Fortran-contiguous
 * array.
 *
 * @tparam T The type of the values
 */
template<typename T>
class FortranContiguousView final : public FortranContiguousConstView<T> {

    public:

        /**
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         * @param array     A pointer to a Fortran-contiguous array of template type `T` that stores the values, the
         *                  view provides access to
         */
        FortranContiguousView(uint32 numRows, uint32 numCols, T* array);

        /**
         * An iterator that provides access to the values in the view and allows to modify them.
         */
        typedef T* iterator;

        /**
         * Returns an `iterator` to the beginning of a specific column.
         *
         * @param col   The column
         * @return      An `iterator` to the beginning
         */
        iterator column_begin(uint32 col);

        /**
         * Returns an `iterator` to the end of a specific column.
         *
         * @param col   The column
         * @return      An `iterator` to the end
         */
        iterator column_end(uint32 col);

};
