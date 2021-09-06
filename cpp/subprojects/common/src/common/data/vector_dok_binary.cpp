#include "common/data/vector_dok_binary.hpp"


BinaryDokVector::index_const_iterator BinaryDokVector::indices_cbegin() const {
    return data_.cbegin();
}

BinaryDokVector::index_const_iterator BinaryDokVector::indices_cend() const {
    return data_.cend();
}

bool BinaryDokVector::operator[](uint32 pos) const {
    return data_.find(pos) != data_.end();
}

void BinaryDokVector::set(uint32 pos, bool value) {
    if (value) {
        data_.insert(pos);
    } else {
        data_.erase(pos);
    }
}

void BinaryDokVector::clear() {
    data_.clear();
}
