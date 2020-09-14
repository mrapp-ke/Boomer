#include "sparse.h"


void BinaryDokMatrix::addValue(uint32 row, uint32 column) {
    data_.insert(std::make_pair(row, column));
}

uint8 BinaryDokMatrix::getValue(uint32 row, uint32 column) {
    return data_.find(std::make_pair(row, column)) != data_.end();
}
