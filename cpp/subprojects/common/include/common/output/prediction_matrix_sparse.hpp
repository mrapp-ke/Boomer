/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/matrix_lil.hpp"
#include <memory>


/**
 * A sparse matrix that provides read-only access to predictions.
 *
 * @tparam T The type of the predictions
 */
template<typename T>
class SparsePredictionMatrix final {

    private:

        std::unique_ptr<LilMatrix<T>> matrixPtr_;

        uint32 numCols_;

        uint32 numNonZeroElements_;

    public:

        /**
         * @param matrixPtr             An unique pointer to an object of type `LilMatrix` that stores the predictions
         * @param numCols               The number of columns in the matrix
         * @param numNonZeroElements    The number of non-zero elements in the matrix
         */
        SparsePredictionMatrix(std::unique_ptr<LilMatrix<T>> matrixPtr, uint32 numCols, uint32 numNonZeroElements);

        /**
         * An iterator that provides read-only access to the elements at a row.
         */
        typedef typename LilMatrix<T>::const_iterator const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the beginning
         */
        const_iterator row_cbegin(uint32 row) const;

        /**
         * Returns a `const_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the end
         */
        const_iterator row_cend(uint32 row) const;

        /**
         * Returns the number of rows in the matrix.
         *
         * @return The number of rows
         */
        uint32 getNumRows() const;

        /**
         * Returns the number of columns in the matrix.
         *
         * @return The number of columns
         */
        uint32 getNumCols() const;

        /**
         * Returns the number of non-zero elements in the matrix.
         *
         * @return The number of non-zero elements
         */
        uint32 getNumNonZeroElements() const;

};
