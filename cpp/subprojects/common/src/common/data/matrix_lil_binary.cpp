#include "common/data/matrix_lil_binary.hpp"


BinaryLilMatrix::BinaryLilMatrix(uint32 numRows)
    : numRows_(numRows), array_(new Row[numRows] {}) {

}

BinaryLilMatrix::~BinaryLilMatrix() {
    delete[] array_;
}

typename BinaryLilMatrix::iterator BinaryLilMatrix::row_begin(uint32 row) {
    Row& rowRef = array_[row];
    return rowRef.begin();
}

typename BinaryLilMatrix::iterator BinaryLilMatrix::row_end(uint32 row) {
    Row& rowRef = array_[row];
    return rowRef.end();
}

typename BinaryLilMatrix::const_iterator BinaryLilMatrix::row_cbegin(uint32 row) const {
    const Row& rowRef = array_[row];
    return rowRef.cbegin();
}

typename BinaryLilMatrix::const_iterator BinaryLilMatrix::row_cend(uint32 row) const {
    const Row& rowRef = array_[row];
    return rowRef.cend();
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
        BinarySparseListVector& vector = array_[i];
        vector.clear();
    }
}
