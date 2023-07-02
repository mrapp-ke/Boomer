#include "boosting/data/statistic_view_example_wise_dense.hpp"

#include "boosting/data/arrays.hpp"
#include "common/data/arrays.hpp"

namespace boosting {

    DenseExampleWiseStatisticConstView::DenseExampleWiseStatisticConstView(uint32 numRows, uint32 numGradients,
                                                                           uint32 numHessians, float64* gradients,
                                                                           float64* hessians)
        : numRows_(numRows), numGradients_(numGradients), numHessians_(numHessians), gradients_(gradients),
          hessians_(hessians) {}

    DenseExampleWiseStatisticConstView::gradient_const_iterator DenseExampleWiseStatisticConstView::gradients_cbegin(
      uint32 row) const {
        return &gradients_[row * numGradients_];
    }

    DenseExampleWiseStatisticConstView::gradient_const_iterator DenseExampleWiseStatisticConstView::gradients_cend(
      uint32 row) const {
        return &gradients_[(row + 1) * numGradients_];
    }

    DenseExampleWiseStatisticConstView::hessian_const_iterator DenseExampleWiseStatisticConstView::hessians_cbegin(
      uint32 row) const {
        return &hessians_[row * numHessians_];
    }

    DenseExampleWiseStatisticConstView::hessian_const_iterator DenseExampleWiseStatisticConstView::hessians_cend(
      uint32 row) const {
        return &hessians_[(row + 1) * numHessians_];
    }

    DenseExampleWiseStatisticConstView::hessian_diagonal_const_iterator
      DenseExampleWiseStatisticConstView::hessians_diagonal_cbegin(uint32 row) const {
        return DiagonalConstIterator<float64>(&hessians_[row * numHessians_], 0);
    }

    DenseExampleWiseStatisticConstView::hessian_diagonal_const_iterator
      DenseExampleWiseStatisticConstView::hessians_diagonal_cend(uint32 row) const {
        return DiagonalConstIterator<float64>(&hessians_[row * numHessians_], numGradients_);
    }

    uint32 DenseExampleWiseStatisticConstView::getNumRows() const {
        return numRows_;
    }

    uint32 DenseExampleWiseStatisticConstView::getNumCols() const {
        return numGradients_;
    }

    DenseExampleWiseStatisticView::DenseExampleWiseStatisticView(uint32 numRows, uint32 numGradients,
                                                                 uint32 numHessians, float64* gradients,
                                                                 float64* hessians)
        : DenseExampleWiseStatisticConstView(numRows, numGradients, numHessians, gradients, hessians) {}

    DenseExampleWiseStatisticView::gradient_iterator DenseExampleWiseStatisticView::gradients_begin(uint32 row) {
        return &gradients_[row * numGradients_];
    }

    DenseExampleWiseStatisticView::gradient_iterator DenseExampleWiseStatisticView::gradients_end(uint32 row) {
        return &gradients_[(row + 1) * numGradients_];
    }

    DenseExampleWiseStatisticView::hessian_iterator DenseExampleWiseStatisticView::hessians_begin(uint32 row) {
        return &hessians_[row * numHessians_];
    }

    DenseExampleWiseStatisticView::hessian_iterator DenseExampleWiseStatisticView::hessians_end(uint32 row) {
        return &hessians_[(row + 1) * numHessians_];
    }

    void DenseExampleWiseStatisticView::clear() {
        setArrayToZeros(gradients_, numRows_ * numGradients_);
        setArrayToZeros(hessians_, numRows_ * numHessians_);
    }

    void DenseExampleWiseStatisticView::addToRow(uint32 row, gradient_const_iterator gradientsBegin,
                                                 gradient_const_iterator gradientsEnd,
                                                 hessian_const_iterator hessiansBegin,
                                                 hessian_const_iterator hessiansEnd, float64 weight) {
        addToArray(&gradients_[row * numGradients_], gradientsBegin, numGradients_, weight);
        addToArray(&hessians_[row * numHessians_], hessiansBegin, numHessians_, weight);
    }

}
