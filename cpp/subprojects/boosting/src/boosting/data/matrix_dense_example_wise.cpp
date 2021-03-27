#include "boosting/data/matrix_dense_example_wise.hpp"
#include "boosting/math/math.hpp"
#include "common/data/arrays.hpp"
#include <cstdlib>


namespace boosting {

    DenseExampleWiseStatisticMatrix::DenseExampleWiseStatisticMatrix(uint32 numRows, uint32 numGradients)
        : DenseExampleWiseStatisticMatrix(numRows, numGradients, false) {

    }

    DenseExampleWiseStatisticMatrix::DenseExampleWiseStatisticMatrix(uint32 numRows, uint32 numGradients, bool init)
        : numRows_(numRows), numGradients_(numGradients), numHessians_(triangularNumber(numGradients)),
          gradients_((float64*) (init ? calloc(numRows * numGradients, sizeof(float64))
                                      : malloc(numRows * numGradients * sizeof(float64)))),
          hessians_((float64*) (init ? calloc(numRows * numHessians_, sizeof(float64))
                                     : malloc(numRows * numHessians_ * sizeof(float64)))) {

    }

    DenseExampleWiseStatisticMatrix::~DenseExampleWiseStatisticMatrix() {
        free(gradients_);
        free(hessians_);
    }

    DenseExampleWiseStatisticMatrix::gradient_iterator DenseExampleWiseStatisticMatrix::gradients_row_begin(
            uint32 row) {
        return &gradients_[row * numGradients_];
    }

    DenseExampleWiseStatisticMatrix::gradient_iterator DenseExampleWiseStatisticMatrix::gradients_row_end(uint32 row) {
        return &gradients_[(row + 1) * numGradients_];
    }

    DenseExampleWiseStatisticMatrix::gradient_const_iterator DenseExampleWiseStatisticMatrix::gradients_row_cbegin(
            uint32 row) const {
        return &gradients_[row * numGradients_];
    }

    DenseExampleWiseStatisticMatrix::gradient_const_iterator DenseExampleWiseStatisticMatrix::gradients_row_cend(
            uint32 row) const {
        return &gradients_[(row + 1) * numGradients_];
    }

    DenseExampleWiseStatisticMatrix::hessian_iterator DenseExampleWiseStatisticMatrix::hessians_row_begin(uint32 row) {
        return &hessians_[row * numHessians_];
    }

    DenseExampleWiseStatisticMatrix::hessian_iterator DenseExampleWiseStatisticMatrix::hessians_row_end(uint32 row) {
        return &hessians_[(row + 1) * numHessians_];
    }

    DenseExampleWiseStatisticMatrix::hessian_const_iterator DenseExampleWiseStatisticMatrix::hessians_row_cbegin(
            uint32 row) const {
        return &hessians_[row * numHessians_];
    }

    DenseExampleWiseStatisticMatrix::hessian_const_iterator DenseExampleWiseStatisticMatrix::hessians_row_cend(
            uint32 row) const {
        return &hessians_[(row + 1) * numHessians_];
    }

    uint32 DenseExampleWiseStatisticMatrix::getNumRows() const {
        return numRows_;
    }

    uint32 DenseExampleWiseStatisticMatrix::getNumCols() const {
        return numGradients_;
    }

    void DenseExampleWiseStatisticMatrix::setAllToZero() {
        setArrayToZeros(gradients_, numRows_ * numGradients_);
        setArrayToZeros(hessians_, numRows_ * numHessians_);
    }

    void DenseExampleWiseStatisticMatrix::addToRow(uint32 row, gradient_const_iterator gradientsBegin,
                                                   gradient_const_iterator gradientsEnd,
                                                   hessian_const_iterator hessiansBegin,
                                                   hessian_const_iterator hessiansEnd, float64 weight) {
        uint32 offset = row * numGradients_;

        for (uint32 i = 0; i < numGradients_; i++) {
            gradients_[offset + i] += (gradientsBegin[i] * weight);
        }

        offset = row * numHessians_;

        for (uint32 i = 0; i < numHessians_; i++) {
            hessians_[offset + i] += (hessiansBegin[i] * weight);
        }
    }

}
