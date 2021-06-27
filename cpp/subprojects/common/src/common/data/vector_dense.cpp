#include "common/data/vector_dense.hpp"
#include <cstdlib>


template<class T>
DenseVector<T>::DenseVector(uint32 numElements)
    : DenseVector<T>(numElements, false) {

}

template<class T>
DenseVector<T>::DenseVector(uint32 numElements, bool init)
    : array_((T*) (init ? calloc(numElements, sizeof(T)) : malloc(numElements * sizeof(T)))),
      numElements_(numElements), maxCapacity_(numElements) {

}

template<class T>
DenseVector<T>::~DenseVector() {
    free(array_);
}

template<class T>
uint32 DenseVector<T>::getNumElements() const {
    return numElements_;
}

template<class T>
T DenseVector<T>::getValue(uint32 pos) const {
    return array_[pos];
}

template<class T>
void DenseVector<T>::setValue(uint32 pos, T value) {
    array_[pos] = value;
}

template<class T>
typename DenseVector<T>::iterator DenseVector<T>::begin() {
    return array_;
}

template<class T>
typename DenseVector<T>::iterator DenseVector<T>::end() {
    return &array_[numElements_];
}

template<class T>
typename DenseVector<T>::const_iterator DenseVector<T>::cbegin() const {
    return array_;
}

template<class T>
typename DenseVector<T>::const_iterator DenseVector<T>::cend() const {
    return &array_[numElements_];
}

template<class T>
void DenseVector<T>::setNumElements(uint32 numElements, bool freeMemory) {
    if (numElements < maxCapacity_) {
        if (freeMemory) {
            array_ = (T*) realloc(array_, numElements * sizeof(T));
            maxCapacity_ = numElements;
        }
    } else if (numElements > maxCapacity_) {
        array_ = (T*) realloc(array_, numElements * sizeof(T));
        maxCapacity_ = numElements;
    }

    numElements_ = numElements;
}

template class DenseVector<uint8>;
template class DenseVector<uint32>;
template class DenseVector<float32>;
template class DenseVector<float64>;
