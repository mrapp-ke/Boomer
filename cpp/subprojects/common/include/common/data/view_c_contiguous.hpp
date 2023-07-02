/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_two_dimensional.hpp"

/**
 * Implements row-wise read-only access to the values that are stored in a pre-allocated C-contiguous array.
 *
 * @tparam T The type of the values
 */
template<typename T>
class MLRLCOMMON_API CContiguousConstView : virtual public ITwoDimensionalView {
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

        virtual ~CContiguousConstView() override {};

        /**
         * An iterator that provides read-only access to the elements in the view.
         */
        typedef const T* value_const_iterator;

        /**
         * Returns a `value_const_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `value_const_iterator` to the beginning of the given row
         */
        value_const_iterator values_cbegin(uint32 row) const;

        /**
         * Returns a `value_const_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `value_const_iterator` to the end of the given row
         */
        value_const_iterator values_cend(uint32 row) const;

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
 * Implements row-wise read and write access to the values that are stored in a pre-allocated C-contiguous array.
 *
 * @tparam T The type of the values
 */
template<typename T>
class MLRLCOMMON_API CContiguousView : public CContiguousConstView<T> {
    public:

        /**
         * @param numRows   The number of rows in the view
         * @param numCols   The number of columns in the view
         * @param array     A pointer to a C-contiguous array of template type `T` that stores the values, the view
         *                  provides access to
         */
        CContiguousView(uint32 numRows, uint32 numCols, T* array);

        virtual ~CContiguousView() override {};

        /**
         * An iterator that provides access to the elements in the view and allows to modify them.
         */
        typedef T* value_iterator;

        /**
         * Returns a `value_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `value_iterator` to the beginning of the given row
         */
        value_iterator values_begin(uint32 row);

        /**
         * Returns a `value_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `value_iterator` to the end of the given row
         */
        value_iterator values_end(uint32 row);
};
