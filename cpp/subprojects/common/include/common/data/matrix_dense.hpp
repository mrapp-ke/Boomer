/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"


/**
 * A two-dimensional matrix that provides random access to a fixed number of elements stored in a C-contiguous array.
 *
 * @tparam T The type of the data that is stored in the matrix
 */
template<class T>
class DenseMatrix : public CContiguousView<T> {

    public:

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         */
        DenseMatrix(uint32 numRows, uint32 numCols);

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         * @param init      True, if all elements in the matrix should be value-initialized, false otherwise
         */
        DenseMatrix(uint32 numRows, uint32 numCols, bool init);

        virtual ~DenseMatrix();

};
