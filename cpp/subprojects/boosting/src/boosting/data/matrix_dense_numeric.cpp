#include "boosting/data/matrix_dense_numeric.hpp"


namespace boosting {

    template<class T>
    DenseNumericMatrix<T>::DenseNumericMatrix(uint32 numRows, uint32 numCols)
        : DenseMatrix<T>(numRows, numCols) {

    }

    template<class T>
    DenseNumericMatrix<T>::DenseNumericMatrix(uint32 numRows, uint32 numCols, bool init)
        : DenseMatrix<T>(numRows, numCols, init) {

    }

    template<class T>
    void DenseNumericMatrix<T>::addToRowFromSubset(uint32 row, typename DenseVector<T>::const_iterator begin,
                                                   typename DenseVector<T>::const_iterator end,
                                                   FullIndexVector::const_iterator indicesBegin,
                                                   FullIndexVector::const_iterator indicesEnd) {
        uint32 offset = row * DenseMatrix<T>::numCols_;

        for (uint32 i = 0; i < DenseMatrix<T>::numCols_; i++) {
            DenseMatrix<T>::array_[offset + i] += begin[i];
        }
    }

    template<class T>
    void DenseNumericMatrix<T>::addToRowFromSubset(uint32 row, typename DenseVector<T>::const_iterator begin,
                                                   typename DenseVector<T>::const_iterator end,
                                                   PartialIndexVector::const_iterator indicesBegin,
                                                   PartialIndexVector::const_iterator indicesEnd) {
        uint32 offset = row * DenseMatrix<T>::numCols_;
        typename DenseVector<T>::const_iterator valueIterator = begin;

        for (auto indexIterator = indicesBegin; indexIterator != indicesEnd; indexIterator++) {
            uint32 index = *indexIterator;
            DenseMatrix<T>::array_[offset + index] += *valueIterator;
            valueIterator++;
        }
    }

    template class DenseNumericMatrix<float64>;

}
