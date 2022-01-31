#include "common/data/view_csr_binary.hpp"


BinaryCsrConstView::BinaryCsrConstView(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices)
    : numRows_(numRows), numCols_(numCols), rowIndices_(rowIndices), colIndices_(colIndices) {

}

BinaryCsrConstView::index_const_iterator BinaryCsrConstView::row_indices_cbegin(uint32 row) const {
    return &colIndices_[rowIndices_[row]];
}

BinaryCsrConstView::index_const_iterator BinaryCsrConstView::row_indices_cend(uint32 row) const {
    return &colIndices_[rowIndices_[row + 1]];
}

BinaryCsrConstView::value_const_iterator BinaryCsrConstView::row_values_cbegin(uint32 row) const {
    return make_binary_forward_iterator(this->row_indices_cbegin(row), this->row_indices_cend(row));
}

BinaryCsrConstView::value_const_iterator BinaryCsrConstView::row_values_cend(uint32 row) const {
    return make_binary_forward_iterator(this->row_indices_cbegin(row), this->row_indices_cend(row), numCols_);
}

uint32 BinaryCsrConstView::getNumNonZeroElements() const {
    return rowIndices_[numRows_];
}

uint32 BinaryCsrConstView::getNumRows() const {
    return numRows_;
}

uint32 BinaryCsrConstView::getNumCols() const {
    return numCols_;
}

BinaryCsrView::BinaryCsrView(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices)
    : BinaryCsrConstView(numRows, numCols, rowIndices, colIndices) {

}

BinaryCsrView::index_iterator BinaryCsrView::row_indices_begin(uint32 row) {
    return &BinaryCsrConstView::colIndices_[BinaryCsrConstView::rowIndices_[row]];
}

BinaryCsrView::index_iterator BinaryCsrView::row_indices_end(uint32 row) {
    return &BinaryCsrConstView::colIndices_[BinaryCsrConstView::rowIndices_[row + 1]];
}
