#include "common/data/vector_sparse_array.hpp"
#include <algorithm>


template<typename T>
SparseArrayVector<T>::IndexConstIterator::IndexConstIterator(
        typename VectorConstView<IndexedValue<T>>::const_iterator iterator)
    : iterator_(iterator) {

}

template<typename T>
typename SparseArrayVector<T>::IndexConstIterator::reference SparseArrayVector<T>::IndexConstIterator::operator[](
        uint32 index) const {
    return iterator_[index].index;
}

template<typename T>
typename SparseArrayVector<T>::IndexConstIterator::reference SparseArrayVector<T>::IndexConstIterator::operator*() const {
    return (*iterator_).index;
}

template<typename T>
typename SparseArrayVector<T>::IndexConstIterator& SparseArrayVector<T>::IndexConstIterator::operator++() {
    ++iterator_;
    return *this;
}

template<typename T>
typename SparseArrayVector<T>::IndexConstIterator& SparseArrayVector<T>::IndexConstIterator::operator++(int n) {
    iterator_++;
    return *this;
}

template<typename T>
typename SparseArrayVector<T>::IndexConstIterator& SparseArrayVector<T>::IndexConstIterator::operator--() {
    --iterator_;
    return *this;
}

template<typename T>
typename SparseArrayVector<T>::IndexConstIterator& SparseArrayVector<T>::IndexConstIterator::operator--(int n) {
    iterator_--;
    return *this;
}

template<typename T>
bool SparseArrayVector<T>::IndexConstIterator::operator!=(const IndexConstIterator& rhs) const {
    return iterator_ != rhs.iterator_;
}

template<typename T>
bool SparseArrayVector<T>::IndexConstIterator::operator==(const IndexConstIterator& rhs) const {
    return iterator_ == rhs.iterator_;
}

template<typename T>
typename SparseArrayVector<T>::IndexConstIterator::difference_type SparseArrayVector<T>::IndexConstIterator::operator-(
        const IndexConstIterator& rhs) const {
    return iterator_ - rhs.iterator_;
}

template<typename T>
SparseArrayVector<T>::IndexIterator::IndexIterator(typename VectorView<IndexedValue<T>>::iterator iterator)
    : iterator_(iterator) {

}

template<typename T>
typename SparseArrayVector<T>::IndexIterator::reference SparseArrayVector<T>::IndexIterator::operator[](
        uint32 index) const {
    return iterator_[index].index;
}

template<typename T>
typename SparseArrayVector<T>::IndexIterator::reference SparseArrayVector<T>::IndexIterator::operator*() const {
    return (*iterator_).index;
}

template<typename T>
typename SparseArrayVector<T>::IndexIterator& SparseArrayVector<T>::IndexIterator::operator++() {
    ++iterator_;
    return *this;
}

template<typename T>
typename SparseArrayVector<T>::IndexIterator& SparseArrayVector<T>::IndexIterator::operator++(int n) {
    iterator_++;
    return *this;
}

template<typename T>
typename SparseArrayVector<T>::IndexIterator& SparseArrayVector<T>::IndexIterator::operator--() {
    --iterator_;
    return *this;
}

template<typename T>
typename SparseArrayVector<T>::IndexIterator& SparseArrayVector<T>::IndexIterator::operator--(int n) {
    iterator_--;
    return *this;
}

template<typename T>
bool SparseArrayVector<T>::IndexIterator::operator!=(const IndexIterator& rhs) const {
    return iterator_ != rhs.iterator_;
}

template<typename T>
bool SparseArrayVector<T>::IndexIterator::operator==(const IndexIterator& rhs) const {
    return iterator_ == rhs.iterator_;
}

template<typename T>
typename SparseArrayVector<T>::IndexIterator::difference_type SparseArrayVector<T>::IndexIterator::operator-(
        const IndexIterator& rhs) const {
    return iterator_ - rhs.iterator_;
}

template<typename T>
SparseArrayVector<T>::ValueConstIterator::ValueConstIterator(
        typename VectorConstView<IndexedValue<T>>::const_iterator iterator)
    : iterator_(iterator) {

}

template<typename T>
typename SparseArrayVector<T>::ValueConstIterator::reference SparseArrayVector<T>::ValueConstIterator::operator[](
        uint32 index) const {
    return iterator_[index].value;
}

template<typename T>
typename SparseArrayVector<T>::ValueConstIterator::reference SparseArrayVector<T>::ValueConstIterator::operator*() const {
    return (*iterator_).value;
}

template<typename T>
typename SparseArrayVector<T>::ValueConstIterator& SparseArrayVector<T>::ValueConstIterator::operator++() {
    ++iterator_;
    return *this;
}

template<typename T>
typename SparseArrayVector<T>::ValueConstIterator& SparseArrayVector<T>::ValueConstIterator::operator++(int n) {
    iterator_++;
    return *this;
}

template<typename T>
typename SparseArrayVector<T>::ValueConstIterator& SparseArrayVector<T>::ValueConstIterator::operator--() {
    --iterator_;
    return *this;
}

template<typename T>
typename SparseArrayVector<T>::ValueConstIterator& SparseArrayVector<T>::ValueConstIterator::operator--(int n) {
    iterator_--;
    return *this;
}

template<typename T>
bool SparseArrayVector<T>::ValueConstIterator::operator!=(const ValueConstIterator& rhs) const {
    return iterator_ != rhs.iterator_;
}

template<typename T>
bool SparseArrayVector<T>::ValueConstIterator::operator==(const ValueConstIterator& rhs) const {
    return iterator_ == rhs.iterator_;
}

template<typename T>
typename SparseArrayVector<T>::ValueConstIterator::difference_type SparseArrayVector<T>::ValueConstIterator::operator-(
        const ValueConstIterator& rhs) const {
    return iterator_ - rhs.iterator_;
}


template<typename T>
SparseArrayVector<T>::ValueIterator::ValueIterator(typename VectorView<IndexedValue<T>>::iterator iterator)
    : iterator_(iterator) {

}

template<typename T>
typename SparseArrayVector<T>::ValueIterator::reference SparseArrayVector<T>::ValueIterator::operator[](
        uint32 index) const {
    return iterator_[index].value;
}

template<typename T>
typename SparseArrayVector<T>::ValueIterator::reference SparseArrayVector<T>::ValueIterator::operator*() const {
    return (*iterator_).value;
}

template<typename T>
typename SparseArrayVector<T>::ValueIterator& SparseArrayVector<T>::ValueIterator::operator++() {
    ++iterator_;
    return *this;
}

template<typename T>
typename SparseArrayVector<T>::ValueIterator& SparseArrayVector<T>::ValueIterator::operator++(int n) {
    iterator_++;
    return *this;
}

template<typename T>
typename SparseArrayVector<T>::ValueIterator& SparseArrayVector<T>::ValueIterator::operator--() {
    --iterator_;
    return *this;
}

template<typename T>
typename SparseArrayVector<T>::ValueIterator& SparseArrayVector<T>::ValueIterator::operator--(int n) {
    iterator_--;
    return *this;
}

template<typename T>
bool SparseArrayVector<T>::ValueIterator::operator!=(const ValueIterator& rhs) const {
    return iterator_ != rhs.iterator_;
}

template<typename T>
bool SparseArrayVector<T>::ValueIterator::operator==(const ValueIterator& rhs) const {
    return iterator_ == rhs.iterator_;
}

template<typename T>
typename SparseArrayVector<T>::ValueIterator::difference_type SparseArrayVector<T>::ValueIterator::operator-(
        const ValueIterator& rhs) const {
    return iterator_ - rhs.iterator_;
}

template<typename T>
SparseArrayVector<T>::SparseArrayVector(uint32 numElements)
    : DenseVector<IndexedValue<T>>(numElements) {

}

template<typename T>
typename SparseArrayVector<T>::index_iterator SparseArrayVector<T>::indices_begin() {
    return IndexIterator(this->begin());
}

template<typename T>
typename SparseArrayVector<T>::index_iterator SparseArrayVector<T>::indices_end() {
    return IndexIterator(this->end());
}

template<typename T>
typename SparseArrayVector<T>::index_const_iterator SparseArrayVector<T>::indices_cbegin() const {
    return IndexConstIterator(this->cbegin());
}

template<typename T>
typename SparseArrayVector<T>::index_const_iterator SparseArrayVector<T>::indices_cend() const {
    return IndexConstIterator(this->cend());
}

template<typename T>
typename SparseArrayVector<T>::value_iterator SparseArrayVector<T>::values_begin() {
    return ValueIterator(this->begin());
}

template<typename T>
typename SparseArrayVector<T>::value_iterator SparseArrayVector<T>::values_end() {
    return ValueIterator(this->end());
}

template<typename T>
typename SparseArrayVector<T>::value_const_iterator SparseArrayVector<T>::values_cbegin() const {
    return ValueConstIterator(this->cbegin());
}

template<typename T>
typename SparseArrayVector<T>::value_const_iterator SparseArrayVector<T>::values_cend() const {
    return ValueConstIterator(this->cend());
}

template<typename T>
void SparseArrayVector<T>::sortByValues() {
    std::sort(this->begin(), this->end(), [=](const IndexedValue<T>& a, const IndexedValue<T>& b) {
        return a.value < b.value;
    });
}

template class SparseArrayVector<uint8>;
template class SparseArrayVector<uint32>;
template class SparseArrayVector<float32>;
template class SparseArrayVector<float64>;
