#include "common/input/label_matrix_dok.hpp"


DokLabelMatrix::DokLabelMatrix(uint32 numRows, uint32 numCols)
    : numRows_(numRows), numCols_(numCols) {

}

uint32 DokLabelMatrix::getNumRows() const {
    return numRows_;
}

uint32 DokLabelMatrix::getNumCols() const {
    return numCols_;
}

uint8 DokLabelMatrix::getValue(uint32 exampleIndex, uint32 labelIndex) const {
    return matrix_.getValue(exampleIndex, labelIndex);
}

void DokLabelMatrix::setValue(uint32 exampleIndex, uint32 labelIndex) {
    matrix_.setValue(exampleIndex, labelIndex);
}
