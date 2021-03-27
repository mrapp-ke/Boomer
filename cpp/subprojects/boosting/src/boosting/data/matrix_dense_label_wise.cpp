#include "boosting/data/matrix_dense_label_wise.hpp"
#include "common/data/arrays.hpp"
#include <cstdlib>


namespace boosting {

    DenseLabelWiseStatisticMatrix::DenseLabelWiseStatisticMatrix(uint32 numRows, uint32 numCols)
        : DenseLabelWiseStatisticMatrix(numRows, numCols, false) {

    }

    DenseLabelWiseStatisticMatrix::DenseLabelWiseStatisticMatrix(uint32 numRows, uint32 numCols, bool init)
        : numRows_(numRows), numCols_(numCols),
          gradients_((float64*) (init ? calloc(numRows * numCols, sizeof(float64))
                                      : malloc(numRows * numCols * sizeof(float64)))),
          hessians_((float64*) (init ? calloc(numRows * numCols, sizeof(float64))
                                     : malloc(numRows * numCols * sizeof(float64)))) {

    }

    DenseLabelWiseStatisticMatrix::~DenseLabelWiseStatisticMatrix() {
        free(gradients_);
        free(hessians_);
    }

    DenseLabelWiseStatisticMatrix::gradient_iterator DenseLabelWiseStatisticMatrix::gradients_row_begin(uint32 row) {
        return &gradients_[row * numCols_];
    }

    DenseLabelWiseStatisticMatrix::gradient_iterator DenseLabelWiseStatisticMatrix::gradients_row_end(uint32 row) {
        return &gradients_[(row + 1) * numCols_];
    }

    DenseLabelWiseStatisticMatrix::gradient_const_iterator DenseLabelWiseStatisticMatrix::gradients_row_cbegin(
            uint32 row) const {
        return &gradients_[row * numCols_];
    }

    DenseLabelWiseStatisticMatrix::gradient_const_iterator DenseLabelWiseStatisticMatrix::gradients_row_cend(
            uint32 row) const {
        return &gradients_[(row + 1) * numCols_];
    }

    DenseLabelWiseStatisticMatrix::hessian_iterator DenseLabelWiseStatisticMatrix::hessians_row_begin(uint32 row) {
        return &hessians_[row * numCols_];
    }

    DenseLabelWiseStatisticMatrix::hessian_iterator DenseLabelWiseStatisticMatrix::hessians_row_end(uint32 row) {
        return &hessians_[(row + 1) * numCols_];
    }

    DenseLabelWiseStatisticMatrix::hessian_const_iterator DenseLabelWiseStatisticMatrix::hessians_row_cbegin(
            uint32 row) const {
        return &hessians_[row * numCols_];
    }

    DenseLabelWiseStatisticMatrix::hessian_const_iterator DenseLabelWiseStatisticMatrix::hessians_row_cend(
            uint32 row) const {
        return &hessians_[(row + 1) * numCols_];
    }

    uint32 DenseLabelWiseStatisticMatrix::getNumRows() const {
        return numRows_;
    }

    uint32 DenseLabelWiseStatisticMatrix::getNumCols() const {
        return numCols_;
    }

    void DenseLabelWiseStatisticMatrix::setAllToZero() {
        setArrayToZeros(gradients_, numRows_ * numCols_);
        setArrayToZeros(hessians_, numRows_ * numCols_);
    }

    void DenseLabelWiseStatisticMatrix::addToRow(uint32 row, gradient_const_iterator gradientsBegin,
                                                 gradient_const_iterator gradientsEnd,
                                                 hessian_const_iterator hessiansBegin,
                                                 hessian_const_iterator hessiansEnd, float64 weight) {
        uint32 offset = row * numCols_;

        for (uint32 i = 0; i < numCols_; i++) {
            uint32 index = offset + i;
            gradients_[index] += (gradientsBegin[i] * weight);
            hessians_[index] += (hessiansBegin[i] * weight);
        }
    }

}
