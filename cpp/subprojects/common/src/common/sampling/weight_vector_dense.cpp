#include "common/sampling/weight_vector_dense.hpp"

#include "common/thresholds/thresholds.hpp"
#include "common/thresholds/thresholds_subset.hpp"

template<typename T>
DenseWeightVector<T>::DenseWeightVector(uint32 numElements) : DenseWeightVector<T>(numElements, false) {}

template<typename T>
DenseWeightVector<T>::DenseWeightVector(uint32 numElements, bool init)
    : vector_(DenseVector<T>(numElements, init)), numNonZeroWeights_(0) {}

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
const T& DenseWeightVector<T>::operator[](uint32 pos) const {
    return vector_[pos];
}

template<typename T>
T& DenseWeightVector<T>::operator[](uint32 pos) {
    return vector_[pos];
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
std::unique_ptr<IThresholdsSubset> DenseWeightVector<T>::createThresholdsSubset(IThresholds& thresholds) const {
    return thresholds.createSubset(*this);
}

template class DenseWeightVector<uint32>;
