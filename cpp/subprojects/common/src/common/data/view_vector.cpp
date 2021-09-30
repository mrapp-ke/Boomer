#include "common/data/view_vector.hpp"
#include "common/data/indexed_value.hpp"


template<typename T>
VectorConstView<T>::VectorConstView(uint32 numElements, T* array)
    : numElements_(numElements), array_(array) {

}

template<typename T>
typename VectorConstView<T>::const_iterator VectorConstView<T>::cbegin() const {
    return array_;
}

template<typename T>
typename VectorConstView<T>::const_iterator VectorConstView<T>::cend() const {
    return &array_[numElements_];
}

template<typename T>
const T& VectorConstView<T>::operator[](uint32 pos) const {
    return array_[pos];
}

template<typename T>
uint32 VectorConstView<T>::getNumElements() const {
    return numElements_;
}

template class VectorConstView<uint8>;
template class VectorConstView<const uint8>;
template class VectorConstView<uint32>;
template class VectorConstView<const uint32>;
template class VectorConstView<float32>;
template class VectorConstView<const float32>;
template class VectorConstView<float64>;
template class VectorConstView<const float64>;
template class VectorConstView<IndexedValue<uint8>>;
template class VectorConstView<IndexedValue<uint32>>;
template class VectorConstView<IndexedValue<float32>>;
template class VectorConstView<IndexedValue<float64>>;

template<typename T>
VectorView<T>::VectorView(uint32 numElements, T* array)
    : VectorConstView<T>(numElements, array) {

}

template<typename T>
typename VectorView<T>::iterator VectorView<T>::begin() {
    return VectorConstView<T>::array_;
}

template<typename T>
typename VectorView<T>::iterator VectorView<T>::end() {
    return &VectorConstView<T>::array_[VectorConstView<T>::numElements_];
}

template<typename T>
const T& VectorView<T>::operator[](uint32 pos) const {
    return VectorConstView<T>::array_[pos];
}

template<typename T>
T& VectorView<T>::operator[](uint32 pos) {
    return VectorConstView<T>::array_[pos];
}

template class VectorView<uint8>;
template class VectorView<uint32>;
template class VectorView<float32>;
template class VectorView<float64>;
template class VectorView<IndexedValue<uint8>>;
template class VectorView<IndexedValue<uint32>>;
template class VectorView<IndexedValue<float32>>;
template class VectorView<IndexedValue<float64>>;
