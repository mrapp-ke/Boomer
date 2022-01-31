/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/matrix_lil_binary.hpp"
#include "common/data/view_csr_binary.hpp"
#include <memory>


/**
 * A sparse matrix that provides read-only access to binary predictions that are stored in the compressed sparse row
 * (CSR) format.
 *
 * The matrix maintains two arrays, referred to as `rowIndices` and `colIndices`. The latter stores a column-index for
 * each of the `numNonZeroValues` non-zero elements in the matrix. The former stores `numRows + 1` row-indices that
 * specify the first element in `colIndices` that correspond to a certain row. The index at the last position is equal
 * to the number of non-zero values in the matrix.
 */
class MLRLCOMMON_API BinarySparsePredictionMatrix final : public BinaryCsrConstView {

    private:

        uint32* rowIndices_;

        uint32* colIndices_;

    public:

        /**
         * @param numRows       The number of rows in the matrix
         * @param numCols       The number of columns in the matrix
         * @param rowIndices    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices
         *                      of the first element in `colIndices` that corresponds to a certain row. The index at the
         *                      last position is equal to `numNonZeroValues`
         * @param colIndices    A pointer to an array of type `uint32`, shape `(numNonZeroValues)`, that stores the
         *                      column-indices, the non-zero elements correspond to
         */
        BinarySparsePredictionMatrix(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices);

        ~BinarySparsePredictionMatrix();

        /**
         * Releases the ownership of the array `rowIndices`. The caller is responsible for freeing the memory that is
         * occupied by the array.
         *
         * @return A pointer to the array `rowIndices`
         */
        uint32* releaseRowIndices();

        /**
         * Releases the ownership of the array `colIndices`. The caller is responsible for freeing the memory that is
         * occupied by the array.
         *
         * @return A pointer to the array `colIndices`
         */
        uint32* releaseColIndices();

};

/**
 * Creates and returns a new object of the type `BinarySparsePredictionMatrix` as a copy of an existing
 * `BinaryLilMatrix`.
 *
 * @param lilMatrix             A reference to an object of type `BinaryLilMatrix` to be copied
 * @param numCols               The number of columns of the given `BinaryLilMatrix`
 * @param numNonZeroElements    The number of non-zero elements in the given `BinaryLilMatrix`
 * @return                      An unique pointer to an object of type `BinarySparsePredictionMatrix` that has been
 *                              created
 */
std::unique_ptr<BinarySparsePredictionMatrix> createBinarySparsePredictionMatrix(const BinaryLilMatrix& lilMatrix,
                                                                                 uint32 numCols,
                                                                                 uint32 numNonZeroElements);
