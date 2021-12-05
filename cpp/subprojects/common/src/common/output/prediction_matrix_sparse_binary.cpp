#include "common/output/prediction_matrix_sparse_binary.hpp"


BinarySparsePredictionMatrix::BinarySparsePredictionMatrix(std::unique_ptr<BinaryLilMatrix> matrixPtr, uint32 numCols,
                                                           uint32 numNonZeroElements)
    : matrixPtr_(std::move(matrixPtr)), numCols_(numCols), numNonZeroElements_(numNonZeroElements) {

}

typename BinarySparsePredictionMatrix::const_iterator BinarySparsePredictionMatrix::row_cbegin(uint32 row) const {
    return matrixPtr_->row_cbegin(row);
}

typename BinarySparsePredictionMatrix::const_iterator BinarySparsePredictionMatrix::row_cend(uint32 row) const {
    return matrixPtr_->row_cend(row);
}

uint32 BinarySparsePredictionMatrix::getNumRows() const {
    return matrixPtr_->getNumRows();
}

uint32 BinarySparsePredictionMatrix::getNumCols() const {
    return numCols_;
}

uint32 BinarySparsePredictionMatrix::getNumNonZeroElements() const {
    return numNonZeroElements_;
}
