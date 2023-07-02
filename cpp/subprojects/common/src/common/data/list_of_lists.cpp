#include "common/data/list_of_lists.hpp"

#include "common/data/indexed_value.hpp"
#include "common/data/triple.hpp"
#include "common/data/tuple.hpp"

template<typename T>
ListOfLists<T>::ListOfLists(uint32 numRows)
    : numRows_(numRows), array_(new std::vector<T>[numRows] {
      }) {}

template<typename T>
ListOfLists<T>::~ListOfLists() {
    delete[] array_;
}

template<typename T>
typename ListOfLists<T>::iterator ListOfLists<T>::begin(uint32 row) {
    return array_[row].begin();
}

template<typename T>
typename ListOfLists<T>::iterator ListOfLists<T>::end(uint32 row) {
    return array_[row].end();
}

template<typename T>
typename ListOfLists<T>::const_iterator ListOfLists<T>::cbegin(uint32 row) const {
    return array_[row].cbegin();
}

template<typename T>
typename ListOfLists<T>::const_iterator ListOfLists<T>::cend(uint32 row) const {
    return array_[row].cend();
}

template<typename T>
typename ListOfLists<T>::row ListOfLists<T>::operator[](uint32 row) {
    return array_[row];
}

template<typename T>
typename ListOfLists<T>::const_row ListOfLists<T>::operator[](uint32 row) const {
    return array_[row];
}

template<typename T>
uint32 ListOfLists<T>::getNumRows() const {
    return numRows_;
}

template<typename T>
void ListOfLists<T>::clear() {
    for (uint32 i = 0; i < numRows_; i++) {
        array_[i].clear();
    }
}

template class ListOfLists<uint8>;
template class ListOfLists<uint32>;
template class ListOfLists<float32>;
template class ListOfLists<float64>;
template class ListOfLists<IndexedValue<uint8>>;
template class ListOfLists<IndexedValue<uint32>>;
template class ListOfLists<IndexedValue<float32>>;
template class ListOfLists<IndexedValue<float64>>;
template class ListOfLists<Tuple<uint8>>;
template class ListOfLists<Tuple<uint32>>;
template class ListOfLists<Tuple<float32>>;
template class ListOfLists<Tuple<float64>>;
template class ListOfLists<IndexedValue<Tuple<uint8>>>;
template class ListOfLists<IndexedValue<Tuple<uint32>>>;
template class ListOfLists<IndexedValue<Tuple<float32>>>;
template class ListOfLists<IndexedValue<Tuple<float64>>>;
template class ListOfLists<Triple<uint8>>;
template class ListOfLists<Triple<uint32>>;
template class ListOfLists<Triple<float32>>;
template class ListOfLists<Triple<float64>>;
template class ListOfLists<IndexedValue<Triple<uint8>>>;
template class ListOfLists<IndexedValue<Triple<uint32>>>;
template class ListOfLists<IndexedValue<Triple<float32>>>;
template class ListOfLists<IndexedValue<Triple<float64>>>;
