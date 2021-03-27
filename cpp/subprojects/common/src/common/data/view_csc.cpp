#include "common/data/view_csc.hpp"


template<class T>
CscView<T>::CscView(uint32 numRows, uint32 numCols, const T* data, const uint32* rowIndices, const uint32* colIndices)
    : numRows_(numRows), numCols_(numCols), data_(data), rowIndices_(rowIndices), colIndices_(colIndices) {

}

template<class T>
typename CscView<T>::value_const_iterator CscView<T>::column_values_cbegin(uint32 col) const {
    return &data_[colIndices_[col]];
}

template<class T>
typename CscView<T>::value_const_iterator CscView<T>::column_values_cend(uint32 col) const {
    return &data_[colIndices_[col + 1]];
}

template<class T>
typename CscView<T>::index_const_iterator CscView<T>::column_indices_cbegin(uint32 col) const {
    return &rowIndices_[colIndices_[col]];
}

template<class T>
typename CscView<T>::index_const_iterator CscView<T>::column_indices_cend(uint32 col) const {
    return &rowIndices_[colIndices_[col + 1]];
}

template<class T>
uint32 CscView<T>::getNumRows() const {
    return numRows_;
}

template<class T>
uint32 CscView<T>::getNumCols() const {
    return numCols_;
}

template<class T>
uint32 CscView<T>::getNumNonZeroElements(uint32 col) const {
    return colIndices_[col + 1] - colIndices_[col];
}

template class CscView<float32>;
