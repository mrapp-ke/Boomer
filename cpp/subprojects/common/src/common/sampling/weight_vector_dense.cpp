#include "common/sampling/weight_vector_dense.hpp"


template<typename T>
DenseWeightVector<T>::DenseWeightVector(uint32 numElements)
    : DenseWeightVector<T>(numElements, false) {

}

template<typename T>
DenseWeightVector<T>::DenseWeightVector(uint32 numElements, bool init)
    : vector_(DenseVector<T>(numElements, init)), numNonZeroWeights_(0) {

}

template<typename T>
typename DenseWeightVector<T>::iterator DenseWeightVector<T>::begin() {
    return vector_.begin();
}

template<typename T>
typename DenseWeightVector<T>::iterator DenseWeightVector<T>::end() {
    return vector_.end();
}

template<typename T>
typename DenseWeightVector<T>::const_iterator DenseWeightVector<T>::cbegin() const {
    return vector_.cbegin();
}

template<typename T>
typename DenseWeightVector<T>::const_iterator DenseWeightVector<T>::cend() const {
    return vector_.cend();
}

template<typename T>
uint32 DenseWeightVector<T>::getNumElements() const {
    return vector_.getNumElements();
}

template<typename T>
uint32 DenseWeightVector<T>::getNumNonZeroWeights() const {
    return numNonZeroWeights_;
}

template<typename T>
void DenseWeightVector<T>::setNumNonZeroWeights(uint32 numNonZeroWeights) {
    numNonZeroWeights_ = numNonZeroWeights;
}

template<typename T>
bool DenseWeightVector<T>::hasZeroWeights() const {
    return numNonZeroWeights_ < vector_.getNumElements();
}

template<typename T>
float64 DenseWeightVector<T>::getWeight(uint32 pos) const {
    return (float64) vector_[pos];
}

template class DenseWeightVector<uint32>;
template class DenseWeightVector<float64>;
