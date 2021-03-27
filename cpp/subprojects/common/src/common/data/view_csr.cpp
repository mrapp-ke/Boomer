#include "common/data/view_csr.hpp"


template<class T>
CsrView<T>::CsrView(uint32 numRows, uint32 numCols, const T* data, const uint32* rowIndices, const uint32* colIndices)
    : numRows_(numRows), numCols_(numCols), data_(data), rowIndices_(rowIndices), colIndices_(colIndices) {

}

template<class T>
typename CsrView<T>::value_const_iterator CsrView<T>::row_values_cbegin(uint32 row) const {
    return &data_[rowIndices_[row]];
}

template<class T>
typename CsrView<T>::value_const_iterator CsrView<T>::row_values_cend(uint32 row) const {
    return &data_[rowIndices_[row + 1]];
}

template<class T>
typename CsrView<T>::index_const_iterator CsrView<T>::row_indices_cbegin(uint32 row) const {
    return &colIndices_[rowIndices_[row]];
}

template<class T>
typename CsrView<T>::index_const_iterator CsrView<T>::row_indices_cend(uint32 row) const {
    return &colIndices_[rowIndices_[row + 1]];
}

template<class T>
uint32 CsrView<T>::getNumRows() const {
    return numRows_;
}

template<class T>
uint32 CsrView<T>::getNumCols() const {
    return numCols_;
}

template<class T>
uint32 CsrView<T>::getNumNonZeroElements(uint32 row) const {
    return rowIndices_[row + 1] - rowIndices_[row];
}

template class CsrView<float32>;
