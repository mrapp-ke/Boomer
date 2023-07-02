/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_two_dimensional.hpp"

/**
 * Implements column-wise read-only access to the values that are stored in a pre-allocated Fortran-contiguous array.
 *
 * @tparam T The type of the values
 */
template<typename T>
class FortranContiguousConstView : virtual public ITwoDimensionalView {
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

        virtual ~FortranContiguousConstView() override {};

        /**
         * An iterator that provides read-only access to the values in the view.
         */
        typedef const T* value_const_iterator;

        /**
         * Returns a `value_const_iterator` to the beginning of a specific column.
         *
         * @param col   The column
         * @return      A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin(uint32 col) const;

        /**
         * Returns a `value_const_iterator` to the end of a specific column.
         *
         * @param col   The column
         * @return      A `value_const_iterator` to the end
         */
        value_const_iterator values_cend(uint32 col) const;

        /**
         * @see `ITwoDimensionalView::getNumRows`
         */
        uint32 getNumRows() const override final;

        /**
         * @see `ITwoDimensionalView::getNumCols`
         */
        uint32 getNumCols() const override final;
};

/**
 * Implements column-wise read and write access to the values that are stored in a pre-allocated Fortran-contiguous
 * array.
 *
 * @tparam T The type of the values
 */
template<typename T>
class FortranContiguousView : public FortranContiguousConstView<T> {
    public:

        /**
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         * @param array     A pointer to a Fortran-contiguous array of template type `T` that stores the values, the
         *                  view provides access to
         */
        FortranContiguousView(uint32 numRows, uint32 numCols, T* array);

        virtual ~FortranContiguousView() override {};

        /**
         * An iterator that provides access to the values in the view and allows to modify them.
         */
        typedef T* value_iterator;

        /**
         * Returns a `value_iterator` to the beginning of a specific column.
         *
         * @param col   The column
         * @return      A `value_iterator` to the beginning
         */
        value_iterator values_begin(uint32 col);

        /**
         * Returns a `value_iterator` to the end of a specific column.
         *
         * @param col   The column
         * @return      A `value_iterator` to the end
         */
        value_iterator values_end(uint32 col);
};
