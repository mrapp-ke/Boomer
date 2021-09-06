#include "common/output/prediction_matrix_sparse.hpp"


template<typename T>
SparsePredictionMatrix<T>::SparsePredictionMatrix(std::unique_ptr<LilMatrix<T>> matrixPtr, uint32 numCols,
                                                  uint32 numNonZeroElements)
    : matrixPtr_(std::move(matrixPtr)), numCols_(numCols), numNonZeroElements_(numNonZeroElements) {

}

template<typename T>
typename SparsePredictionMatrix<T>::const_iterator SparsePredictionMatrix<T>::row_cbegin(uint32 row) const {
    return matrixPtr_->row_cbegin(row);
}

template<typename T>
typename SparsePredictionMatrix<T>::const_iterator SparsePredictionMatrix<T>::row_cend(uint32 row) const {
    return matrixPtr_->row_cend(row);
}

template<typename T>
uint32 SparsePredictionMatrix<T>::getNumRows() const {
    return matrixPtr_->getNumRows();
}

template<typename T>
uint32 SparsePredictionMatrix<T>::getNumCols() const {
    return numCols_;
}

template<typename T>
uint32 SparsePredictionMatrix<T>::getNumNonZeroElements() const {
    return numNonZeroElements_;
}

template class SparsePredictionMatrix<uint8>;
template class SparsePredictionMatrix<uint32>;
template class SparsePredictionMatrix<float32>;
template class SparsePredictionMatrix<float64>;
