#include "common/data/view_csr_binary.hpp"

BinaryCsrConstView::BinaryCsrConstView(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices)
    : numRows_(numRows), numCols_(numCols), rowIndices_(rowIndices), colIndices_(colIndices) {}

BinaryCsrConstView::index_const_iterator BinaryCsrConstView::indices_cbegin(uint32 row) const {
    return &colIndices_[rowIndices_[row]];
}

BinaryCsrConstView::index_const_iterator BinaryCsrConstView::indices_cend(uint32 row) const {
    return &colIndices_[rowIndices_[row + 1]];
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
    : BinaryCsrConstView(numRows, numCols, rowIndices, colIndices) {}

BinaryCsrView::index_iterator BinaryCsrView::indices_begin(uint32 row) {
    return &BinaryCsrConstView::colIndices_[BinaryCsrConstView::rowIndices_[row]];
}

BinaryCsrView::index_iterator BinaryCsrView::indices_end(uint32 row) {
    return &BinaryCsrConstView::colIndices_[BinaryCsrConstView::rowIndices_[row + 1]];
}
