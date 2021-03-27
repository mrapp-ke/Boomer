#include "boosting/data/vector_dense_example_wise.hpp"
#include "common/data/arrays.hpp"
#include "boosting/math/math.hpp"
#include <cstdlib>


namespace boosting {

    DenseExampleWiseStatisticVector::HessianDiagonalIterator::HessianDiagonalIterator(
            const DenseExampleWiseStatisticVector& vector, uint32 index)
        : vector_(vector), index_(index) {

    }

    DenseExampleWiseStatisticVector::HessianDiagonalIterator::reference DenseExampleWiseStatisticVector::HessianDiagonalIterator::operator[](
            uint32 index) const {
        return vector_.hessians_[triangularNumber(index + 1) - 1];
    }

    DenseExampleWiseStatisticVector::HessianDiagonalIterator::reference DenseExampleWiseStatisticVector::HessianDiagonalIterator::operator*() const {
        return vector_.hessians_[triangularNumber(index_ + 1) - 1];
    }

    DenseExampleWiseStatisticVector::HessianDiagonalIterator& DenseExampleWiseStatisticVector::HessianDiagonalIterator::operator++() {
        ++index_;
        return *this;
    }

    DenseExampleWiseStatisticVector::HessianDiagonalIterator& DenseExampleWiseStatisticVector::HessianDiagonalIterator::operator++(
            int n) {
        index_++;
        return *this;
    }

    DenseExampleWiseStatisticVector::HessianDiagonalIterator& DenseExampleWiseStatisticVector::HessianDiagonalIterator::operator--() {
        --index_;
        return *this;
    }

    DenseExampleWiseStatisticVector::HessianDiagonalIterator& DenseExampleWiseStatisticVector::HessianDiagonalIterator::operator--(
            int n) {
        index_--;
        return *this;
    }

    bool DenseExampleWiseStatisticVector::HessianDiagonalIterator::operator!=(
            const DenseExampleWiseStatisticVector::HessianDiagonalIterator& rhs) const {
        return index_ != rhs.index_;
    }

    DenseExampleWiseStatisticVector::HessianDiagonalIterator::difference_type DenseExampleWiseStatisticVector::HessianDiagonalIterator::operator-(
            const DenseExampleWiseStatisticVector::HessianDiagonalIterator& rhs) const {
        return (difference_type) index_ - (difference_type) rhs.index_;
    }

    DenseExampleWiseStatisticVector::DenseExampleWiseStatisticVector(uint32 numGradients)
        : DenseExampleWiseStatisticVector(numGradients, false) {

    }

    DenseExampleWiseStatisticVector::DenseExampleWiseStatisticVector(uint32 numGradients, bool init)
        : numGradients_(numGradients), numHessians_(triangularNumber(numGradients)),
          gradients_((float64*) (init ? calloc(numGradients, sizeof(float64))
                                      : malloc(numGradients * sizeof(float64)))),
          hessians_((float64*) (init ? calloc(numHessians_, sizeof(float64))
                                      : malloc(numHessians_ * sizeof(float64)))) {

    }

    DenseExampleWiseStatisticVector::DenseExampleWiseStatisticVector(const DenseExampleWiseStatisticVector& vector)
        : DenseExampleWiseStatisticVector(vector.numGradients_) {
        copyArray(vector.gradients_, gradients_, numGradients_);
        copyArray(vector.hessians_, hessians_, numHessians_);
    }

    DenseExampleWiseStatisticVector::~DenseExampleWiseStatisticVector() {
        free(gradients_);
        free(hessians_);
    }

    DenseExampleWiseStatisticVector::gradient_iterator DenseExampleWiseStatisticVector::gradients_begin() {
        return gradients_;
    }

    DenseExampleWiseStatisticVector::gradient_iterator DenseExampleWiseStatisticVector::gradients_end() {
        return &gradients_[numGradients_];
    }

    DenseExampleWiseStatisticVector::gradient_const_iterator DenseExampleWiseStatisticVector::gradients_cbegin() const {
        return gradients_;
    }

    DenseExampleWiseStatisticVector::gradient_const_iterator DenseExampleWiseStatisticVector::gradients_cend() const {
        return &gradients_[numGradients_];
    }

    DenseExampleWiseStatisticVector::hessian_iterator DenseExampleWiseStatisticVector::hessians_begin() {
        return hessians_;
    }

    DenseExampleWiseStatisticVector::hessian_iterator DenseExampleWiseStatisticVector::hessians_end() {
        return &hessians_[numHessians_];
    }

    DenseExampleWiseStatisticVector::hessian_const_iterator DenseExampleWiseStatisticVector::hessians_cbegin() const {
        return hessians_;
    }

    DenseExampleWiseStatisticVector::hessian_const_iterator DenseExampleWiseStatisticVector::hessians_cend() const {
        return &hessians_[numHessians_];
    }

    DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator DenseExampleWiseStatisticVector::hessians_diagonal_cbegin() const  {
        return DenseExampleWiseStatisticVector::HessianDiagonalIterator(*this, 0);
    }

    DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator DenseExampleWiseStatisticVector::hessians_diagonal_cend() const  {
        return DenseExampleWiseStatisticVector::HessianDiagonalIterator(*this, numGradients_);
    }

    uint32 DenseExampleWiseStatisticVector::getNumElements() const {
        return numGradients_;
    }

    void DenseExampleWiseStatisticVector::setAllToZero() {
        setArrayToZeros(gradients_, numGradients_);
        setArrayToZeros(hessians_, numHessians_);
    }

    void DenseExampleWiseStatisticVector::add(gradient_const_iterator gradientsBegin,
                                              gradient_const_iterator gradientsEnd,
                                              hessian_const_iterator hessiansBegin,
                                              hessian_const_iterator hessiansEnd) {
        for (uint32 i = 0; i < numGradients_; i++) {
            gradients_[i] += gradientsBegin[i];
        }

        for (uint32 i = 0; i < numHessians_; i++) {
            hessians_[i] += hessiansBegin[i];
        }
    }

    void DenseExampleWiseStatisticVector::add(gradient_const_iterator gradientsBegin,
                                              gradient_const_iterator gradientsEnd,
                                              hessian_const_iterator hessiansBegin,
                                              hessian_const_iterator hessiansEnd, float64 weight) {
        for (uint32 i = 0; i < numGradients_; i++) {
            gradients_[i] += (gradientsBegin[i] * weight);
        }

        for (uint32 i = 0; i < numHessians_; i++) {
            hessians_[i] += (hessiansBegin[i] * weight);
        }
    }

    void DenseExampleWiseStatisticVector::subtract(gradient_const_iterator gradientsBegin,
                                                   gradient_const_iterator gradientsEnd,
                                                   hessian_const_iterator hessiansBegin,
                                                   hessian_const_iterator hessiansEnd, float64 weight) {
        for (uint32 i = 0; i < numGradients_; i++) {
            gradients_[i] -= (gradientsBegin[i] * weight);
        }

        for (uint32 i = 0; i < numHessians_; i++) {
            hessians_[i] -= (hessiansBegin[i] * weight);
        }
    }

    void DenseExampleWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                      gradient_const_iterator gradientsEnd,
                                                      hessian_const_iterator hessiansBegin,
                                                      hessian_const_iterator hessiansEnd,
                                                      const FullIndexVector& indices, float64 weight) {
        for (uint32 i = 0; i < numGradients_; i++) {
            gradients_[i] += (gradientsBegin[i] * weight);
        }

        for (uint32 i = 0; i < numHessians_; i++) {
            hessians_[i] += (hessiansBegin[i] * weight);
        }
    }

    void DenseExampleWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                      gradient_const_iterator gradientsEnd,
                                                      hessian_const_iterator hessiansBegin,
                                                      hessian_const_iterator hessiansEnd,
                                                      const PartialIndexVector& indices, float64 weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        hessian_iterator hessianIterator = hessians_;

        for (uint32 i = 0; i < numGradients_; i++) {
            uint32 index = indexIterator[i];
            gradients_[i] += (gradientsBegin[index] * weight);
            uint32 offset = triangularNumber(index);

            for (uint32 j = 0; j < i + 1; j++) {
                uint32 index2 = indexIterator[j];
                *hessianIterator += (weight * hessiansBegin[offset + index2]);
                hessianIterator++;
            }
        }
    }

    void DenseExampleWiseStatisticVector::difference(gradient_const_iterator firstGradientsBegin,
                                                     gradient_const_iterator firstGradientsEnd,
                                                     hessian_const_iterator firstHessiansBegin,
                                                     hessian_const_iterator firstHessiansEnd,
                                                     const FullIndexVector& firstIndices,
                                                     gradient_const_iterator secondGradientsBegin,
                                                     gradient_const_iterator secondGradientsEnd,
                                                     hessian_const_iterator secondHessiansBegin,
                                                     hessian_const_iterator secondHessiansEnd) {
        for (uint32 i = 0; i < numGradients_; i++) {
            gradients_[i] = firstGradientsBegin[i] - secondGradientsBegin[i];
        }

        for (uint32 i = 0; i < numHessians_; i++) {
            hessians_[i] = firstHessiansBegin[i] - secondHessiansBegin[i];
        }
    }

    void DenseExampleWiseStatisticVector::difference(gradient_const_iterator firstGradientsBegin,
                                                     gradient_const_iterator firstGradientsEnd,
                                                     hessian_const_iterator firstHessiansBegin,
                                                     hessian_const_iterator firstHessiansEnd,
                                                     const PartialIndexVector& firstIndices,
                                                     gradient_const_iterator secondGradientsBegin,
                                                     gradient_const_iterator secondGradientsEnd,
                                                     hessian_const_iterator secondHessiansBegin,
                                                     hessian_const_iterator secondHessiansEnd) {
        PartialIndexVector::const_iterator firstIndexIterator = firstIndices.cbegin();
        hessian_iterator hessianIterator = hessians_;
        hessian_const_iterator secondHessianIterator = secondHessiansBegin;

        for (uint32 i = 0; i < numGradients_; i++) {
            uint32 firstIndex = firstIndexIterator[i];
            gradients_[i] = firstGradientsBegin[firstIndex] - secondGradientsBegin[i];
            uint32 offset = triangularNumber(firstIndex);

            for (uint32 j = 0; j < i + 1; j++) {
                uint32 firstIndex2 = firstIndexIterator[j];
                *hessianIterator = firstHessiansBegin[offset + firstIndex2] - *secondHessianIterator;
                hessianIterator++;
                secondHessianIterator++;
            }
        }
    }

}
