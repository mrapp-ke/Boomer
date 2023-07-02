/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/matrix_sparse_set.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"

namespace boosting {

    /**
     * A two-dimensional matrix that provides row-wise access to values that are stored in the list of lists (LIL)
     * format.
     *
     * @tparam T The type of the values that are stored in the matrix
     */
    template<typename T>
    class NumericSparseSetMatrix final : public SparseSetMatrix<T> {
        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            NumericSparseSetMatrix(uint32 numRows, uint32 numCols);

            /**
             * Adds all values in another vector to certain elements, whose positions are given as a
             * `CompleteIndexVector`, at a specific row of this matrix.
             *
             * @param row           The row
             * @param begin         An iterator to the beginning of the vector
             * @param end           An iterator to the end of the vector
             * @param indicesBegin  An iterator to the beginning of the indices
             * @param indicesEnd    An iterator to the end of the indices
             */
            void addToRowFromSubset(uint32 row, typename VectorConstView<T>::const_iterator begin,
                                    typename VectorConstView<T>::const_iterator end,
                                    CompleteIndexVector::const_iterator indicesBegin,
                                    CompleteIndexVector::const_iterator indicesEnd);

            /**
             * Adds all values in another vector to certain elements, whose positions are given as a
             * `PartialIndexVector`, at a specific row of this matrix.
             *
             * @param row           The row
             * @param begin         An iterator to the beginning of the vector
             * @param end           An iterator to the end of the vector
             * @param indicesBegin  An iterator to the beginning of the indices
             * @param indicesEnd    An iterator to the end of the indices
             */
            void addToRowFromSubset(uint32 row, typename VectorConstView<T>::const_iterator begin,
                                    typename VectorConstView<T>::const_iterator end,
                                    PartialIndexVector::const_iterator indicesBegin,
                                    PartialIndexVector::const_iterator indicesEnd);

            /**
             * Subtracts all values in another vector from certain elements, whose positions are given as a
             * `CompleteIndexVector`, at a specific row of this matrix.
             *
             * @param row           The row
             * @param begin         An iterator to the beginning of the vector
             * @param end           An iterator to the end of the vector
             * @param indicesBegin  An iterator to the beginning of the indices
             * @param indicesEnd    An iterator to the end of the indices
             */
            void removeFromRowFromSubset(uint32 row, typename VectorConstView<T>::const_iterator begin,
                                         typename VectorConstView<T>::const_iterator end,
                                         CompleteIndexVector::const_iterator indicesBegin,
                                         CompleteIndexVector::const_iterator indicesEnd);

            /**
             * Subtracts all values in another vector from certain elements, whose positions are given as a
             * `PartialIndexVector`, at a specific row of this matrix.
             *
             * @param row           The row
             * @param begin         An iterator to the beginning of the vector
             * @param end           An iterator to the end of the vector
             * @param indicesBegin  An iterator to the beginning of the indices
             * @param indicesEnd    An iterator to the end of the indices
             */
            void removeFromRowFromSubset(uint32 row, typename VectorConstView<T>::const_iterator begin,
                                         typename VectorConstView<T>::const_iterator end,
                                         PartialIndexVector::const_iterator indicesBegin,
                                         PartialIndexVector::const_iterator indicesEnd);
    };

}
