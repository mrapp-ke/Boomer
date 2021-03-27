#include "common/data/matrix_dense.hpp"
#include <cstdlib>


template<class T>
DenseMatrix<T>::DenseMatrix(uint32 numRows, uint32 numCols)
    : DenseMatrix<T>(numRows, numCols, false) {

}

template<class T>
DenseMatrix<T>::DenseMatrix(uint32 numRows, uint32 numCols, bool init)
    : CContiguousView<T>(numRows, numCols,
                         (T*) (init ? calloc(numRows * numCols, sizeof(T)) : malloc(numRows * numCols * sizeof(T)))) {

}

template<class T>
DenseMatrix<T>::~DenseMatrix() {
    free(CContiguousView<T>::array_);
}

template class DenseMatrix<float64>;
