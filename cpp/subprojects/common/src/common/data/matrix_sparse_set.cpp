#include "common/data/matrix_sparse_set.hpp"

#include "common/data/arrays.hpp"
#include "common/data/triple.hpp"
#include "common/data/tuple.hpp"

static const uint32 MAX_INDEX = std::numeric_limits<uint32>::max();

template<typename T>
static inline void clearRow(typename LilMatrix<T>::row row,
                            typename CContiguousView<uint32>::value_iterator indexIterator) {
    while (!row.empty()) {
        const IndexedValue<T>& lastEntry = row.back();
        indexIterator[lastEntry.index] = MAX_INDEX;
        row.pop_back();
    }
}

template<typename T>
SparseSetMatrix<T>::ConstRow::ConstRow(typename LilMatrix<T>::const_row row,
                                       typename CContiguousView<uint32>::value_const_iterator indexIterator)
    : row_(row), indexIterator_(indexIterator) {}

template<typename T>
typename LilMatrix<T>::const_iterator SparseSetMatrix<T>::ConstRow::cbegin() const {
    return row_.cbegin();
}

template<typename T>
typename LilMatrix<T>::const_iterator SparseSetMatrix<T>::ConstRow::cend() const {
    return row_.cend();
}

template<typename T>
uint32 SparseSetMatrix<T>::ConstRow::getNumElements() const {
    return (uint32) row_.size();
}

template<typename T>
const IndexedValue<T>* SparseSetMatrix<T>::ConstRow::operator[](uint32 index) const {
    uint32 i = indexIterator_[index];
    return i == MAX_INDEX ? nullptr : &row_[i];
}

template<typename T>
SparseSetMatrix<T>::Row::Row(typename LilMatrix<T>::row row,
                             typename CContiguousView<uint32>::value_iterator indexIterator)
    : row_(row), indexIterator_(indexIterator) {}

template<typename T>
typename LilMatrix<T>::iterator SparseSetMatrix<T>::Row::begin() {
    return row_.begin();
}

template<typename T>
typename LilMatrix<T>::iterator SparseSetMatrix<T>::Row::end() {
    return row_.end();
}

template<typename T>
typename LilMatrix<T>::const_iterator SparseSetMatrix<T>::Row::cbegin() const {
    return row_.cbegin();
}

template<typename T>
typename LilMatrix<T>::const_iterator SparseSetMatrix<T>::Row::cend() const {
    return row_.cend();
}

template<typename T>
uint32 SparseSetMatrix<T>::Row::getNumElements() const {
    return (uint32) row_.size();
}

template<typename T>
const IndexedValue<T>* SparseSetMatrix<T>::Row::operator[](uint32 index) const {
    uint32 i = indexIterator_[index];
    return i == MAX_INDEX ? nullptr : &row_[i];
}

template<typename T>
IndexedValue<T>& SparseSetMatrix<T>::Row::emplace(uint32 index) {
    uint32 i = indexIterator_[index];

    if (i == MAX_INDEX) {
        indexIterator_[index] = (uint32) row_.size();
        row_.emplace_back(index);
        return row_.back();
    }

    return row_[i];
}

template<typename T>
IndexedValue<T>& SparseSetMatrix<T>::Row::emplace(uint32 index, const T& defaultValue) {
    uint32 i = indexIterator_[index];

    if (i == MAX_INDEX) {
        indexIterator_[index] = (uint32) row_.size();
        row_.emplace_back(index, defaultValue);
        return row_.back();
    }

    return row_[i];
}

template<typename T>
void SparseSetMatrix<T>::Row::erase(uint32 index) {
    uint32 i = indexIterator_[index];

    if (i != MAX_INDEX) {
        const IndexedValue<T>& lastEntry = row_.back();
        uint32 lastIndex = lastEntry.index;

        if (lastIndex != index) {
            row_[i] = lastEntry;
            indexIterator_[lastIndex] = i;
        }

        indexIterator_[index] = MAX_INDEX;
        row_.pop_back();
    }
}

template<typename T>
void SparseSetMatrix<T>::Row::clear() {
    clearRow<T>(row_, indexIterator_);
}

template<typename T>
SparseSetMatrix<T>::SparseSetMatrix(uint32 numRows, uint32 numCols)
    : lilMatrix_(LilMatrix<T>(numRows)), indexMatrix_(CContiguousMatrix<uint32>(numRows, numCols)) {
    setArrayToValue(indexMatrix_.values_begin(0), numRows * numCols, MAX_INDEX);
}

template<typename T>
typename SparseSetMatrix<T>::iterator SparseSetMatrix<T>::begin(uint32 row) {
    return lilMatrix_.begin(row);
}

template<typename T>
typename SparseSetMatrix<T>::iterator SparseSetMatrix<T>::end(uint32 row) {
    return lilMatrix_.end(row);
}

template<typename T>
typename SparseSetMatrix<T>::const_iterator SparseSetMatrix<T>::cbegin(uint32 row) const {
    return lilMatrix_.cbegin(row);
}

template<typename T>
typename SparseSetMatrix<T>::const_iterator SparseSetMatrix<T>::cend(uint32 row) const {
    return lilMatrix_.cend(row);
}

template<typename T>
typename SparseSetMatrix<T>::row SparseSetMatrix<T>::operator[](uint32 row) {
    return Row(lilMatrix_[row], indexMatrix_.values_begin(row));
}

template<typename T>
typename SparseSetMatrix<T>::const_row SparseSetMatrix<T>::operator[](uint32 row) const {
    return ConstRow(lilMatrix_[row], indexMatrix_.values_cbegin(row));
}

template<typename T>
uint32 SparseSetMatrix<T>::getNumRows() const {
    return lilMatrix_.getNumRows();
}

template<typename T>
uint32 SparseSetMatrix<T>::getNumCols() const {
    return indexMatrix_.getNumCols();
}

template<typename T>
void SparseSetMatrix<T>::clear() {
    uint32 numRows = lilMatrix_.getNumRows();

    for (uint32 i = 0; i < numRows; i++) {
        clearRow<T>(lilMatrix_[i], indexMatrix_.values_begin(i));
    }
}

template class SparseSetMatrix<uint8>;
template class SparseSetMatrix<uint32>;
template class SparseSetMatrix<float32>;
template class SparseSetMatrix<float64>;
template class SparseSetMatrix<Tuple<uint8>>;
template class SparseSetMatrix<Tuple<uint32>>;
template class SparseSetMatrix<Tuple<float32>>;
template class SparseSetMatrix<Tuple<float64>>;
template class SparseSetMatrix<Triple<uint8>>;
template class SparseSetMatrix<Triple<uint32>>;
template class SparseSetMatrix<Triple<float32>>;
template class SparseSetMatrix<Triple<float64>>;
