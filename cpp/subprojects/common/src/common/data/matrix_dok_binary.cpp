#include "common/data/matrix_dok_binary.hpp"


bool BinaryDokMatrix::getValue(uint32 row, uint32 column) const {
    return data_.find(std::make_pair(row, column)) != data_.end();
}

void BinaryDokMatrix::setValue(uint32 row, uint32 column) {
    data_.insert(std::make_pair(row, column));
}

void BinaryDokMatrix::setAllToZero() {
    data_.clear();
}
