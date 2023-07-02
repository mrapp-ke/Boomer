/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/matrix_c_contiguous.hpp"
#include "common/data/matrix_lil.hpp"

/**
 * A two-dimensional matrix that provides row-wise access to data that is stored in the list of lists (LIL) format. In
 * contrast to a `LilMatrix`, this matrix does also provide random access to its elements. This additional functionality
 * comes at the expense of memory efficiency, as it requires to not only maintain a sparse matrix that stores the
 * non-zero elements, but also a dense matrix that stores for each element the corresponding position in the sparse
 * matrix, if available.
 *
 * The data structure that is used for the representation of a single row is often referred to as an "unordered sparse
 * set". It was originally proposed in "An efficient representation for sparse sets", Briggs, Torczon, 1993 (see
 * https://dl.acm.org/doi/pdf/10.1145/176454.176484).
 *
 * @tparam T The type of the values that are stored in the matrix
 */
template<typename T>
class SparseSetMatrix : virtual public ITwoDimensionalView {
    private:

        /**
         * Provides read-only access to a single row in the matrix.
         */
        class ConstRow final {
            private:

                const typename LilMatrix<T>::const_row row_;

                typename CContiguousView<uint32>::value_const_iterator indexIterator_;

            public:

                /**
                 * @param row           A `LilMatrix::const_row` that provides access to the non-zero elements at the
                 *                      row
                 * @param indexIterator An iterator that provides access to the indices in `row` that correspond to
                 *                      individual columns
                 */
                ConstRow(const typename LilMatrix<T>::const_row row,
                         CContiguousView<uint32>::value_const_iterator indexIterator);

                /**
                 * An iterator that provides read-only access to the elements in the row.
                 */
                typedef typename LilMatrix<T>::const_iterator const_iterator;

                /**
                 * Returns a `const_iterator` to the beginning of the row.
                 *
                 * @return A `const_iterator` to the beginning
                 */
                const_iterator cbegin() const;

                /**
                 * Returns a `const_iterator` to the end of the row.
                 *
                 * @return A `const_iterator` to the end
                 */
                const_iterator cend() const;

                /**
                 * Returns the number of non-zero elements in the row.
                 *
                 * @return The number of non-zero elements in the row
                 */
                uint32 getNumElements() const;

                /**
                 * Returns a pointer to the element that corresponds to a specific index.
                 *
                 * @param index The index of the element to be returned
                 * @return      A pointer to the element that corresponds to the given index or a null pointer, if no
                 *              such element is available
                 */
                const IndexedValue<T>* operator[](uint32 index) const;
        };

        /**
         * Provides access to a single row in the matrix and allows to modify its elements.
         */
        class Row final {
            private:

                const typename LilMatrix<T>::row row_;

                typename CContiguousView<uint32>::value_iterator indexIterator_;

            public:

                /**
                 * @param row           A `LilMatrix::row` that provides access to the the non-zero elements at the row
                 * @param indexIterator An iterator that provides access to the indices in `row` that correspond to
                 *                      individual columns
                 */
                Row(const typename LilMatrix<T>::row row, CContiguousView<uint32>::value_iterator indexIterator);

                /**
                 * Returns a `LilMatrix::iterator` to the beginning of the row.
                 *
                 * @return A `LilMatrix::iterator` to the beginning
                 */
                typename LilMatrix<T>::iterator begin();

                /**
                 * Returns a `LilMatrix::iterator` to the end of the row.
                 *
                 * @return A `LilMatrix::iterator` to the end
                 */
                typename LilMatrix<T>::iterator end();

                /**
                 * Returns a `LilMatrix::const_iterator` to the beginning of the row.
                 *
                 * @return A `LilMatrix::const_iterator` to the beginning
                 */
                typename LilMatrix<T>::const_iterator cbegin() const;

                /**
                 * Returns a `LilMatrix::const_iterator` to the end of the row.
                 *
                 * @return A `LilMatrix::const_iterator` to the end
                 */
                typename LilMatrix<T>::const_iterator cend() const;

                /**
                 * Returns the number of non-zero elements in the row.
                 *
                 * @return The number of non-zero elements in the row
                 */
                uint32 getNumElements() const;

                /**
                 * Returns a pointer to the element that corresponds to a specific index.
                 *
                 * @param index The index of the element to be returned
                 * @return      A pointer to the element that corresponds to the given index or a null pointer, if no
                 *              such element is available
                 */
                const IndexedValue<T>* operator[](uint32 index) const;

                /**
                 * Returns a reference to the element that corresponds to a specific index. If no such element is
                 * available, it is inserted into the vector.
                 *
                 * @param index The index of the element to be returned
                 * @return      A reference to the element that corresponds to the given index
                 */
                IndexedValue<T>& emplace(uint32 index);

                /**
                 * Returns a reference to the element that corresponds to a specific index. If no such element is
                 * available, it is inserted into the vector using a specific default value.
                 *
                 * @param index         The index of the element to be returned
                 * @param defaultValue  The default value to be used
                 * @return              A reference to the element that corresponds to the given index
                 */
                IndexedValue<T>& emplace(uint32 index, const T& defaultValue);

                /**
                 * Removes the element that corresponds to a specific index, if available.
                 *
                 * @param index The index of the element to be removed
                 */
                void erase(uint32 index);

                /**
                 * Removes all elements from the row.
                 */
                void clear();
        };

        LilMatrix<T> lilMatrix_;

        CContiguousMatrix<uint32> indexMatrix_;

    public:

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         */
        SparseSetMatrix(uint32 numRows, uint32 numCols);

        virtual ~SparseSetMatrix() override {};

        /**
         * Provides access to a row and allows to modify its elements.
         */
        typedef typename SparseSetMatrix<T>::Row row;

        /**
         * Provides read-only access to a row.
         */
        typedef typename SparseSetMatrix<T>::ConstRow const_row;

        /**
         * An iterator that provides access to the elements at a row and allows to modify them.
         */
        typedef typename LilMatrix<T>::iterator iterator;

        /**
         * An iterator that provides read-only access to the elements at a row.
         */
        typedef typename LilMatrix<T>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the beginning
         */
        iterator begin(uint32 row);

        /**
         * Returns an `iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the end
         */
        iterator end(uint32 row);

        /**
         * Returns a `const_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the beginning
         */
        const_iterator cbegin(uint32 row) const;

        /**
         * Returns a `const_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the end
         */
        const_iterator cend(uint32 row) const;

        /**
         * Provides access to a specific row and allows to modify its elements.
         *
         * @param row   The index of the row
         * @return      A `row`
         */
        row operator[](uint32 row);

        /**
         * Provides read-only access to a specific row.
         *
         * @param row   The index of the row
         * @return      A `const_row`
         */
        const_row operator[](uint32 row) const;

        /**
         * Sets the values of all elements to zero.
         */
        void clear();

        /**
         * @see `ITwoDimensionalView::getNumRows`
         */
        uint32 getNumRows() const override;

        /**
         * @see `ITwoDimensionalView::getNumCols`
         */
        uint32 getNumCols() const override;
};
