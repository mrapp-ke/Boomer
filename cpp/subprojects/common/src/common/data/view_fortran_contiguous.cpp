#include "common/data/view_fortran_contiguous.hpp"


template<class T>
FortranContiguousView<T>::FortranContiguousView(uint32 numRows, uint32 numCols, T* array)
    : numRows_(numRows), numCols_(numCols), array_(array) {

}

template<class T>
typename FortranContiguousView<T>::iterator FortranContiguousView<T>::column_begin(uint32 col) {
    return &array_[col * numRows_];
}

template<class T>
typename FortranContiguousView<T>::iterator FortranContiguousView<T>::column_end(uint32 col) {
    return &array_[(col + 1) * numRows_];
}

template<class T>
typename FortranContiguousView<T>::const_iterator FortranContiguousView<T>::column_cbegin(uint32 col) const {
    return &array_[col * numRows_];
}

template<class T>
typename FortranContiguousView<T>::const_iterator FortranContiguousView<T>::column_cend(uint32 col) const {
    return &array_[(col + 1) * numRows_];
}

template<class T>
uint32 FortranContiguousView<T>::getNumRows() const {
    return numRows_;
}

template<class T>
uint32 FortranContiguousView<T>::getNumCols() const {
    return numCols_;
}

template class FortranContiguousView<float32>;
