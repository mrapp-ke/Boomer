#include "common/data/vector_sparse_list_binary.hpp"


BinarySparseListVector::index_const_iterator BinarySparseListVector::indices_cbegin() const {
    return indices_.cbegin();
}

BinarySparseListVector::index_const_iterator BinarySparseListVector::indices_cend() const {
    return indices_.cend();
}

uint32 BinarySparseListVector::getNumElements() const {
    return (uint32) indices_.size();
}

void BinarySparseListVector::setValue(uint32 pos) {
    return indices_.push_back(pos);
}

void BinarySparseListVector::setAllToZero() {
    return indices_.clear();
}
