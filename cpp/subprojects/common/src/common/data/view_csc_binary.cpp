#include "common/data/view_csc_binary.hpp"


BinaryCscConstView::BinaryCscConstView(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices)
    : numRows_(numRows), numCols_(numCols), rowIndices_(rowIndices), colIndices_(colIndices) {

}

BinaryCscConstView::index_const_iterator BinaryCscConstView::column_indices_cbegin(uint32 col) const {
    return &rowIndices_[colIndices_[col]];
}

BinaryCscConstView::index_const_iterator BinaryCscConstView::column_indices_cend(uint32 col) const {
    return &rowIndices_[colIndices_[col + 1]];
}

BinaryCscConstView::value_const_iterator BinaryCscConstView::column_values_cbegin(uint32 col) const {
    return make_binary_forward_iterator(this->column_indices_cbegin(col), this->column_indices_cend(col));
}

BinaryCscConstView::value_const_iterator BinaryCscConstView::column_values_cend(uint32 col) const {
    return make_binary_forward_iterator(this->column_indices_cbegin(col), this->column_indices_cend(col), numRows_);
}

uint32 BinaryCscConstView::getNumRows() const {
    return numRows_;
}

uint32 BinaryCscConstView::getNumCols() const {
    return numCols_;
}

uint32 BinaryCscConstView::getNumNonZeroElements() const {
    return colIndices_[numCols_];
}

BinaryCscView::BinaryCscView(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices)
    : BinaryCscConstView(numRows, numCols, rowIndices, colIndices) {

}

BinaryCscView::index_iterator BinaryCscView::column_indices_begin(uint32 col) {
    return &BinaryCscConstView::rowIndices_[BinaryCscConstView::colIndices_[col]];
}

BinaryCscView::index_iterator BinaryCscView::column_indices_end(uint32 col) {
    return &BinaryCscConstView::rowIndices_[BinaryCscConstView::colIndices_[col + 1]];
}
