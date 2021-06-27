#include "common/data/vector_binned_dense.hpp"
#include <cstdlib>


template<class T>
DenseBinnedVector<T>::Iterator::Iterator(const DenseBinnedVector<T>& vector, uint32 index)
    : vector_(vector), index_(index) {

}

template<class T>
typename DenseBinnedVector<T>::Iterator::reference DenseBinnedVector<T>::Iterator::operator[](uint32 index) const {
    uint32 binIndex = vector_.binIndices_[index];
    return vector_.array_[binIndex];
}

template<class T>
typename DenseBinnedVector<T>::Iterator::reference DenseBinnedVector<T>::Iterator::operator*() const {
    uint32 binIndex = vector_.binIndices_[index_];
    return vector_.array_[binIndex];
}

template<class T>
typename DenseBinnedVector<T>::Iterator& DenseBinnedVector<T>::Iterator::operator++() {
    ++index_;
    return *this;
}

template<class T>
typename DenseBinnedVector<T>::Iterator& DenseBinnedVector<T>::Iterator::operator++(int n) {
    index_++;
    return *this;
}

template<class T>
typename DenseBinnedVector<T>::Iterator& DenseBinnedVector<T>::Iterator::operator--() {
    --index_;
    return *this;
}

template<class T>
typename DenseBinnedVector<T>::Iterator& DenseBinnedVector<T>::Iterator::operator--(int n) {
    index_--;
    return *this;
}

template<class T>
bool DenseBinnedVector<T>::Iterator::operator!=(const DenseBinnedVector<T>::Iterator& rhs) const {
    return index_ != rhs.index_;
}

template<class T>
typename DenseBinnedVector<T>::Iterator::difference_type DenseBinnedVector<T>::Iterator::operator-(
        const DenseBinnedVector<T>::Iterator& rhs) const {
    return (difference_type) index_ - (difference_type) rhs.index_;
}

template<class T>
DenseBinnedVector<T>::DenseBinnedVector(uint32 numElements, uint32 numBins)
    : binIndices_((uint32*) malloc(numElements * sizeof(uint32))), array_((T*) malloc(numBins * sizeof(T))),
      numElements_(numElements), numBins_(numBins), maxBinCapacity_(numBins) {

}

template<class T>
DenseBinnedVector<T>::~DenseBinnedVector() {
    free(binIndices_);
    free(array_);
}

template<class T>
typename DenseBinnedVector<T>::const_iterator DenseBinnedVector<T>::cbegin() const {
    return DenseBinnedVector<T>::Iterator(*this, 0);
}

template<class T>
typename DenseBinnedVector<T>::const_iterator DenseBinnedVector<T>::cend() const {
    return DenseBinnedVector<T>::Iterator(*this, numElements_);
}

template<class T>
typename DenseBinnedVector<T>::index_binned_iterator DenseBinnedVector<T>::indices_binned_begin() {
    return binIndices_;
}

template<class T>
typename DenseBinnedVector<T>::index_binned_iterator DenseBinnedVector<T>::indices_binned_end() {
    return &binIndices_[numBins_];
}

template<class T>
typename DenseBinnedVector<T>::index_binned_const_iterator DenseBinnedVector<T>::indices_binned_cbegin() const {
    return binIndices_;
}

template<class T>
typename DenseBinnedVector<T>::index_binned_const_iterator DenseBinnedVector<T>::indices_binned_cend() const {
    return &binIndices_[numBins_];
}

template<class T>
typename DenseBinnedVector<T>::binned_iterator DenseBinnedVector<T>::binned_begin() {
    return array_;
}

template<class T>
typename DenseBinnedVector<T>::binned_iterator DenseBinnedVector<T>::binned_end() {
    return &array_[numBins_];
}

template<class T>
typename DenseBinnedVector<T>::binned_const_iterator DenseBinnedVector<T>::binned_cbegin() const {
    return array_;
}

template<class T>
typename DenseBinnedVector<T>::binned_const_iterator DenseBinnedVector<T>::binned_cend() const {
    return &array_[numBins_];
}

template<class T>
uint32 DenseBinnedVector<T>::getNumElements() const {
    return numElements_;
}

template<class T>
uint32 DenseBinnedVector<T>::getNumBins() const {
    return numBins_;
}

template<class T>
void DenseBinnedVector<T>::setNumBins(uint32 numBins, bool freeMemory) {
    if (numBins < maxBinCapacity_) {
        if (freeMemory) {
            array_ = (T*) realloc(array_, numBins * sizeof(T));
            maxBinCapacity_ = numBins;
        }
    } else if (numBins > maxBinCapacity_) {
        array_ = (T*) realloc(array_, numBins * sizeof(T));
        maxBinCapacity_ = numBins;
    }

    numBins_ = numBins;
}

template<class T>
T DenseBinnedVector<T>::getValue(uint32 pos) const {
    uint32 binIndex = binIndices_[pos];
    return array_[binIndex];
}

template class DenseBinnedVector<float64>;
