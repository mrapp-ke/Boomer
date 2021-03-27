#include "common/data/view_c_contiguous.hpp"


template<class T>
CContiguousView<T>::CContiguousView(uint32 numRows, uint32 numCols, T* array)
    : numRows_(numRows), numCols_(numCols), array_(array) {

}

template<class T>
typename CContiguousView<T>::iterator CContiguousView<T>::row_begin(uint32 row) {
    return &array_[row * numCols_];
}

template<class T>
typename CContiguousView<T>::iterator CContiguousView<T>::row_end(uint32 row) {
    return &array_[(row + 1) * numCols_];
}

template<class T>
typename CContiguousView<T>::const_iterator CContiguousView<T>::row_cbegin(uint32 row) const {
    return &array_[row * numCols_];
}

template<class T>
typename CContiguousView<T>::const_iterator CContiguousView<T>::row_cend(uint32 row) const {
    return &array_[(row + 1) * numCols_];
}

template<class T>
uint32 CContiguousView<T>::getNumRows() const {
    return numRows_;
}

template<class T>
uint32 CContiguousView<T>::getNumCols() const {
    return numCols_;
}

template class CContiguousView<uint8>;
template class CContiguousView<float32>;
template class CContiguousView<float64>;
