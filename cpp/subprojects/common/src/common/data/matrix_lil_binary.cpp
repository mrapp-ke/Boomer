#include "common/data/matrix_lil_binary.hpp"


BinaryLilMatrix::BinaryLilMatrix(uint32 numRows)
    : numRows_(numRows), array_(new Row[numRows] {}) {

}

BinaryLilMatrix::~BinaryLilMatrix() {
    delete[] array_;
}

typename BinaryLilMatrix::iterator BinaryLilMatrix::row_begin(uint32 row) {
    return array_[row].begin();
}

typename BinaryLilMatrix::iterator BinaryLilMatrix::row_end(uint32 row) {
    return array_[row].end();
}

typename BinaryLilMatrix::const_iterator BinaryLilMatrix::row_cbegin(uint32 row) const {
    return array_[row].cbegin();
}

typename BinaryLilMatrix::const_iterator BinaryLilMatrix::row_cend(uint32 row) const {
    return array_[row].cend();
}

typename BinaryLilMatrix::Row& BinaryLilMatrix::getRow(uint32 row) {
    return array_[row];
}

const typename BinaryLilMatrix::Row& BinaryLilMatrix::getRow(uint32 row) const {
    return array_[row];
}

uint32 BinaryLilMatrix::getNumRows() const {
    return numRows_;
}

void BinaryLilMatrix::clear() {
    for (uint32 i = 0; i < numRows_; i++) {
        array_[i].clear();
    }
}
