#include "common/input/label_matrix_c_contiguous.hpp"


CContiguousLabelMatrix::CContiguousLabelMatrix(uint32 numRows, uint32 numCols, uint8* array)
    : view_(CContiguousView<uint8>(numRows, numCols, array)) {

}

uint32 CContiguousLabelMatrix::getNumRows() const {
    return view_.getNumRows();
}

uint32 CContiguousLabelMatrix::getNumCols() const {
    return view_.getNumCols();
}

uint8 CContiguousLabelMatrix::getValue(uint32 exampleIndex, uint32 labelIndex) const {
    return view_.row_cbegin(exampleIndex)[labelIndex];
}
