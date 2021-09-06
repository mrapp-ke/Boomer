#include "common/data/matrix_dense.hpp"
#include <cstdlib>


template<typename T>
DenseMatrix<T>::DenseMatrix(uint32 numRows, uint32 numCols)
    : DenseMatrix<T>(numRows, numCols, false) {

}

template<typename T>
DenseMatrix<T>::DenseMatrix(uint32 numRows, uint32 numCols, bool init)
    : CContiguousView<T>(numRows, numCols,
                         (T*) (init ? calloc(numRows * numCols, sizeof(T)) : malloc(numRows * numCols * sizeof(T)))) {

}

template<typename T>
DenseMatrix<T>::~DenseMatrix() {
    free(this->array_);
}

template class DenseMatrix<uint8>;
template class DenseMatrix<uint32>;
template class DenseMatrix<float32>;
template class DenseMatrix<float64>;
