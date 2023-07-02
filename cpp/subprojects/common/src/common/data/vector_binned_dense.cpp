#include "common/data/vector_binned_dense.hpp"

template<typename T>
DenseBinnedVector<T>::ValueConstIterator::ValueConstIterator(DenseVector<uint32>::const_iterator binIndexIterator,
                                                             typename DenseVector<T>::const_iterator valueIterator)
    : binIndexIterator_(binIndexIterator), valueIterator_(valueIterator) {}

template<typename T>
typename DenseBinnedVector<T>::ValueConstIterator::reference DenseBinnedVector<T>::ValueConstIterator::operator[](
  uint32 index) const {
    uint32 binIndex = binIndexIterator_[index];
    return valueIterator_[binIndex];
}

template<typename T>
typename DenseBinnedVector<T>::ValueConstIterator::reference DenseBinnedVector<T>::ValueConstIterator::operator*()
  const {
    uint32 binIndex = *binIndexIterator_;
    return valueIterator_[binIndex];
}

template<typename T>
typename DenseBinnedVector<T>::ValueConstIterator& DenseBinnedVector<T>::ValueConstIterator::operator++() {
    ++binIndexIterator_;
    return *this;
}

template<typename T>
typename DenseBinnedVector<T>::ValueConstIterator& DenseBinnedVector<T>::ValueConstIterator::operator++(int n) {
    binIndexIterator_++;
    return *this;
}

template<typename T>
typename DenseBinnedVector<T>::ValueConstIterator& DenseBinnedVector<T>::ValueConstIterator::operator--() {
    --binIndexIterator_;
    return *this;
}

template<typename T>
typename DenseBinnedVector<T>::ValueConstIterator& DenseBinnedVector<T>::ValueConstIterator::operator--(int n) {
    binIndexIterator_--;
    return *this;
}

template<typename T>
bool DenseBinnedVector<T>::ValueConstIterator::operator!=(const ValueConstIterator& rhs) const {
    return binIndexIterator_ != rhs.binIndexIterator_;
}

template<typename T>
bool DenseBinnedVector<T>::ValueConstIterator::operator==(const ValueConstIterator& rhs) const {
    return binIndexIterator_ == rhs.binIndexIterator_;
}

template<typename T>
typename DenseBinnedVector<T>::ValueConstIterator::difference_type DenseBinnedVector<T>::ValueConstIterator::operator-(
  const ValueConstIterator& rhs) const {
    return (difference_type) (binIndexIterator_ - rhs.binIndexIterator_);
}

template<typename T>
DenseBinnedVector<T>::DenseBinnedVector(uint32 numElements, uint32 numBins)
    : binIndices_(DenseVector<uint32>(numElements)), values_(DenseVector<T>(numBins)) {}

template<typename T>
typename DenseBinnedVector<T>::const_iterator DenseBinnedVector<T>::cbegin() const {
    return ValueConstIterator(binIndices_.cbegin(), values_.cbegin());
}

template<typename T>
typename DenseBinnedVector<T>::const_iterator DenseBinnedVector<T>::cend() const {
    return ValueConstIterator(binIndices_.cend(), values_.cbegin());
}

template<typename T>
typename DenseBinnedVector<T>::index_iterator DenseBinnedVector<T>::indices_begin() {
    return binIndices_.begin();
}

template<typename T>
typename DenseBinnedVector<T>::index_iterator DenseBinnedVector<T>::indices_end() {
    return binIndices_.end();
}

template<typename T>
typename DenseBinnedVector<T>::index_const_iterator DenseBinnedVector<T>::indices_cbegin() const {
    return binIndices_.cbegin();
}

template<typename T>
typename DenseBinnedVector<T>::index_const_iterator DenseBinnedVector<T>::indices_cend() const {
    return binIndices_.cend();
}

template<typename T>
typename DenseBinnedVector<T>::value_iterator DenseBinnedVector<T>::values_begin() {
    return values_.begin();
}

template<typename T>
typename DenseBinnedVector<T>::value_iterator DenseBinnedVector<T>::values_end() {
    return values_.end();
}

template<typename T>
typename DenseBinnedVector<T>::value_const_iterator DenseBinnedVector<T>::values_cbegin() const {
    return values_.cbegin();
}

template<typename T>
typename DenseBinnedVector<T>::value_const_iterator DenseBinnedVector<T>::values_cend() const {
    return values_.cend();
}

template<typename T>
uint32 DenseBinnedVector<T>::getNumElements() const {
    return binIndices_.getNumElements();
}

template<typename T>
uint32 DenseBinnedVector<T>::getNumBins() const {
    return values_.getNumElements();
}

template<typename T>
void DenseBinnedVector<T>::setNumBins(uint32 numBins, bool freeMemory) {
    values_.setNumElements(numBins, freeMemory);
}

template class DenseBinnedVector<uint8>;
template class DenseBinnedVector<uint32>;
template class DenseBinnedVector<float32>;
template class DenseBinnedVector<float64>;
