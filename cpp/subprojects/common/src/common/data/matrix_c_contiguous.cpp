#include "common/data/matrix_c_contiguous.hpp"

#include <cstdlib>

template<typename T>
CContiguousMatrix<T>::CContiguousMatrix(uint32 numRows, uint32 numCols)
    : CContiguousMatrix<T>(numRows, numCols, false) {}

template<typename T>
CContiguousMatrix<T>::CContiguousMatrix(uint32 numRows, uint32 numCols, bool init)
    : CContiguousView<T>(numRows, numCols,
                         (T*) (init ? calloc(numRows * numCols, sizeof(T)) : malloc(numRows * numCols * sizeof(T)))) {}

template<typename T>
CContiguousMatrix<T>::~CContiguousMatrix() {
    free(this->array_);
}

template class CContiguousMatrix<uint8>;
template class CContiguousMatrix<uint32>;
template class CContiguousMatrix<float32>;
template class CContiguousMatrix<float64>;
