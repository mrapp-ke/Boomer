#include "boosting/data/statistic_vector_example_wise_dense.hpp"
#include "boosting/math/math.hpp"
#include "boosting/data/arrays.hpp"
#include "common/data/arrays.hpp"
#include <cstdlib>


namespace boosting {

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
        return DiagonalConstIterator<float64>(hessians_, 0);
    }

    DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator DenseExampleWiseStatisticVector::hessians_diagonal_cend() const  {
        return DiagonalConstIterator<float64>(hessians_, numGradients_);
    }

    uint32 DenseExampleWiseStatisticVector::getNumElements() const {
        return numGradients_;
    }

    void DenseExampleWiseStatisticVector::clear() {
        setArrayToZeros(gradients_, numGradients_);
        setArrayToZeros(hessians_, numHessians_);
    }

    void DenseExampleWiseStatisticVector::add(gradient_const_iterator gradientsBegin,
                                              gradient_const_iterator gradientsEnd,
                                              hessian_const_iterator hessiansBegin,
                                              hessian_const_iterator hessiansEnd) {
        addToArray(gradients_, gradientsBegin, numGradients_);
        addToArray(hessians_, hessiansBegin, numHessians_);
    }

    void DenseExampleWiseStatisticVector::add(gradient_const_iterator gradientsBegin,
                                              gradient_const_iterator gradientsEnd,
                                              hessian_const_iterator hessiansBegin,
                                              hessian_const_iterator hessiansEnd, float64 weight) {
        addToArray(gradients_, gradientsBegin, numGradients_, weight);
        addToArray(hessians_, hessiansBegin, numHessians_, weight);
    }

    void DenseExampleWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                      gradient_const_iterator gradientsEnd,
                                                      hessian_const_iterator hessiansBegin,
                                                      hessian_const_iterator hessiansEnd,
                                                      const CompleteIndexVector& indices, float64 weight) {
        addToArray(gradients_, gradientsBegin, numGradients_, weight);
        addToArray(hessians_, hessiansBegin, numHessians_, weight);
    }

    void DenseExampleWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                      gradient_const_iterator gradientsEnd,
                                                      hessian_const_iterator hessiansBegin,
                                                      hessian_const_iterator hessiansEnd,
                                                      const PartialIndexVector& indices, float64 weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToArray(gradients_, gradientsBegin, indexIterator, numGradients_, weight);

        for (uint32 i = 0; i < numGradients_; i++) {
            uint32 index = indexIterator[i];
            addToArray(&hessians_[triangularNumber(i)], &hessiansBegin[triangularNumber(index)], indexIterator, i + 1,
                       weight);
        }
    }

    void DenseExampleWiseStatisticVector::difference(gradient_const_iterator firstGradientsBegin,
                                                     gradient_const_iterator firstGradientsEnd,
                                                     hessian_const_iterator firstHessiansBegin,
                                                     hessian_const_iterator firstHessiansEnd,
                                                     const CompleteIndexVector& firstIndices,
                                                     gradient_const_iterator secondGradientsBegin,
                                                     gradient_const_iterator secondGradientsEnd,
                                                     hessian_const_iterator secondHessiansBegin,
                                                     hessian_const_iterator secondHessiansEnd) {
        setArrayToDifference(gradients_, firstGradientsBegin, secondGradientsBegin, numGradients_);
        setArrayToDifference(hessians_, firstHessiansBegin, secondHessiansBegin, numHessians_);
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
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        setArrayToDifference(gradients_, firstGradientsBegin, secondGradientsBegin, indexIterator, numGradients_);

        for (uint32 i = 0; i < numGradients_; i++) {
            uint32 offset = triangularNumber(i);
            uint32 index = indexIterator[i];
            setArrayToDifference(&hessians_[offset], &firstHessiansBegin[triangularNumber(index)],
                                 &secondHessiansBegin[offset], indexIterator, i + 1);
        }
    }

}
