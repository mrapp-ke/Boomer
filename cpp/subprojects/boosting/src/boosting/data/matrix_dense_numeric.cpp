#include "boosting/data/matrix_dense_numeric.hpp"


namespace boosting {

    template<typename T>
    NumericDenseMatrix<T>::NumericDenseMatrix(uint32 numRows, uint32 numCols)
        : DenseMatrix<T>(numRows, numCols) {

    }

    template<typename T>
    NumericDenseMatrix<T>::NumericDenseMatrix(uint32 numRows, uint32 numCols, bool init)
        : DenseMatrix<T>(numRows, numCols, init) {

    }

    template<typename T>
    void NumericDenseMatrix<T>::addToRowFromSubset(uint32 row, typename VectorConstView<T>::const_iterator begin,
                                                   typename VectorConstView<T>::const_iterator end,
                                                   CompleteIndexVector::const_iterator indicesBegin,
                                                   CompleteIndexVector::const_iterator indicesEnd) {
        typename NumericDenseMatrix<T>::value_iterator iterator = this->row_values_begin(row);
        uint32 numCols = this->getNumCols();

        for (uint32 i = 0; i < numCols; i++) {
            iterator[i] += begin[i];
        }
    }

    template<typename T>
    void NumericDenseMatrix<T>::addToRowFromSubset(uint32 row, typename VectorConstView<T>::const_iterator begin,
                                                   typename VectorConstView<T>::const_iterator end,
                                                   PartialIndexVector::const_iterator indicesBegin,
                                                   PartialIndexVector::const_iterator indicesEnd) {
        typename NumericDenseMatrix<T>::value_iterator iterator = this->row_values_begin(row);
        uint32 numCols = indicesEnd - indicesBegin;

        for (uint32 i = 0; i < numCols; i++) {
            uint32 index = indicesBegin[i];
            iterator[index] += begin[i];
        }
    }

    template class NumericDenseMatrix<uint8>;
    template class NumericDenseMatrix<uint32>;
    template class NumericDenseMatrix<float32>;
    template class NumericDenseMatrix<float64>;

}
