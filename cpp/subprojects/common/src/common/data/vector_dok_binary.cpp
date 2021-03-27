#include "common/data/vector_dok_binary.hpp"


BinaryDokVector::index_const_iterator BinaryDokVector::indices_cbegin() const {
    return data_.cbegin();
}

BinaryDokVector::index_const_iterator BinaryDokVector::indices_cend() const {
    return data_.cend();
}

bool BinaryDokVector::getValue(uint32 pos) const {
    return data_.find(pos) != data_.end();
}

void BinaryDokVector::setValue(uint32 pos) {
    data_.insert(pos);
}

void BinaryDokVector::setAllToZero() {
    data_.clear();
}
