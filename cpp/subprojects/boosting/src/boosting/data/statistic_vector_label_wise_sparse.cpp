#include "boosting/data/statistic_vector_label_wise_sparse.hpp"

#include "boosting/data/arrays.hpp"
#include "common/data/arrays.hpp"
#include "statistic_vector_label_wise_sparse_common.hpp"

#include <cstdlib>

namespace boosting {

    SparseLabelWiseStatisticVector::ConstIterator::ConstIterator(const Triple<float64>* iterator, float64 sumOfWeights)
        : iterator_(iterator), sumOfWeights_(sumOfWeights) {}

    SparseLabelWiseStatisticVector::ConstIterator::value_type SparseLabelWiseStatisticVector::ConstIterator::operator[](
      uint32 index) const {
        const Triple<float64>& triple = iterator_[index];
        float64 gradient = triple.first;
        float64 hessian = triple.second + (sumOfWeights_ - triple.third);
        return Tuple<float64>(gradient, hessian);
    }

    SparseLabelWiseStatisticVector::ConstIterator::value_type SparseLabelWiseStatisticVector::ConstIterator::operator*()
      const {
        const Triple<float64>& triple = *iterator_;
        float64 gradient = triple.first;
        float64 hessian = triple.second + (sumOfWeights_ - triple.third);
        return Tuple<float64>(gradient, hessian);
    }

    SparseLabelWiseStatisticVector::ConstIterator& SparseLabelWiseStatisticVector::ConstIterator::operator++() {
        ++iterator_;
        return *this;
    }

    SparseLabelWiseStatisticVector::ConstIterator& SparseLabelWiseStatisticVector::ConstIterator::operator++(int n) {
        iterator_++;
        return *this;
    }

    SparseLabelWiseStatisticVector::ConstIterator& SparseLabelWiseStatisticVector::ConstIterator::operator--() {
        --iterator_;
        return *this;
    }

    SparseLabelWiseStatisticVector::ConstIterator& SparseLabelWiseStatisticVector::ConstIterator::operator--(int n) {
        iterator_--;
        return *this;
    }

    bool SparseLabelWiseStatisticVector::ConstIterator::operator!=(const ConstIterator& rhs) const {
        return iterator_ != rhs.iterator_;
    }

    bool SparseLabelWiseStatisticVector::ConstIterator::operator==(const ConstIterator& rhs) const {
        return iterator_ == rhs.iterator_;
    }

    SparseLabelWiseStatisticVector::ConstIterator::difference_type
      SparseLabelWiseStatisticVector::ConstIterator::operator-(const ConstIterator& rhs) const {
        return iterator_ - rhs.iterator_;
    }

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(uint32 numElements)
        : SparseLabelWiseStatisticVector(numElements, false) {}

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(uint32 numElements, bool init)
        : numElements_(numElements),
          statistics_((Triple<float64>*) (init ? calloc(numElements, sizeof(Triple<float64>))
                                               : malloc(numElements * sizeof(Triple<float64>)))),
          sumOfWeights_(0) {}

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(const SparseLabelWiseStatisticVector& vector)
        : SparseLabelWiseStatisticVector(vector.numElements_) {
        copyArray(vector.statistics_, statistics_, numElements_);
        sumOfWeights_ = vector.sumOfWeights_;
    }

    SparseLabelWiseStatisticVector::~SparseLabelWiseStatisticVector() {
        free(statistics_);
    }

    SparseLabelWiseStatisticVector::const_iterator SparseLabelWiseStatisticVector::cbegin() const {
        return ConstIterator(statistics_, sumOfWeights_);
    }

    SparseLabelWiseStatisticVector::const_iterator SparseLabelWiseStatisticVector::cend() const {
        return ConstIterator(&statistics_[numElements_], sumOfWeights_);
    }

    uint32 SparseLabelWiseStatisticVector::getNumElements() const {
        return numElements_;
    }

    void SparseLabelWiseStatisticVector::clear() {
        sumOfWeights_ = 0;
        setArrayToZeros(statistics_, numElements_);
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticVector& vector) {
        sumOfWeights_ += vector.sumOfWeights_;
        addToArray(statistics_, vector.statistics_, numElements_);
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticConstView& view, uint32 row) {
        sumOfWeights_ += 1;
        addToSparseLabelWiseStatisticVector(statistics_, view.cbegin(row), view.cend(row));
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                             float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            addToSparseLabelWiseStatisticVector(statistics_, view.cbegin(row), view.cend(row), weight);
        }
    }

    void SparseLabelWiseStatisticVector::remove(const SparseLabelWiseStatisticConstView& view, uint32 row) {
        sumOfWeights_ -= 1;
        removeFromSparseLabelWiseStatisticVector(statistics_, view.cbegin(row), view.cend(row));
    }

    void SparseLabelWiseStatisticVector::remove(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                float64 weight) {
        if (weight != 0) {
            sumOfWeights_ -= weight;
            removeFromSparseLabelWiseStatisticVector(statistics_, view.cbegin(row), view.cend(row), weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                     const CompleteIndexVector& indices) {
        sumOfWeights_ += 1;
        addToSparseLabelWiseStatisticVector(statistics_, view.cbegin(row), view.cend(row));
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                     const PartialIndexVector& indices) {
        sumOfWeights_ += 1;
        SparseLabelWiseStatisticConstView::const_row viewRow = view[row];
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        uint32 numElements = indices.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            const IndexedValue<Tuple<float64>>* entry = viewRow[index];

            if (entry) {
                const Tuple<float64>& tuple = entry->value;
                Triple<float64>& triple = statistics_[i];
                triple.first += (tuple.first);
                triple.second += (tuple.second);
                triple.third += 1;
            }
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                     const CompleteIndexVector& indices, float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            addToSparseLabelWiseStatisticVector(statistics_, view.cbegin(row), view.cend(row), weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                     const PartialIndexVector& indices, float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            SparseLabelWiseStatisticConstView::const_row viewRow = view[row];
            PartialIndexVector::const_iterator indexIterator = indices.cbegin();
            uint32 numElements = indices.getNumElements();

            for (uint32 i = 0; i < numElements; i++) {
                uint32 index = indexIterator[i];
                const IndexedValue<Tuple<float64>>* entry = viewRow[index];

                if (entry) {
                    const Tuple<float64>& tuple = entry->value;
                    Triple<float64>& triple = statistics_[i];
                    triple.first += (tuple.first * weight);
                    triple.second += (tuple.second * weight);
                    triple.third += weight;
                }
            }
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                                                     const CompleteIndexVector& indices) {
        SparseLabelWiseHistogramConstView::weight_const_iterator weightIterator = view.weights_cbegin();
        float64 binWeight = weightIterator[row];

        if (binWeight != 0) {
            sumOfWeights_ += binWeight;
            addToArray(statistics_, view.cbegin(row), numElements_);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                                                     const PartialIndexVector& indices) {
        SparseLabelWiseHistogramConstView::weight_const_iterator weightIterator = view.weights_cbegin();
        float64 binWeight = weightIterator[row];

        if (binWeight != 0) {
            sumOfWeights_ += binWeight;
            addToArray(statistics_, view.cbegin(row), indices.cbegin(), indices.getNumElements());
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                                                     const CompleteIndexVector& indices, float64 weight) {
        SparseLabelWiseHistogramConstView::weight_const_iterator weightIterator = view.weights_cbegin();
        float64 binWeight = weightIterator[row] * weight;

        if (binWeight != 0) {
            sumOfWeights_ += binWeight;
            addToArray(statistics_, view.cbegin(row), numElements_, weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                                                     const PartialIndexVector& indices, float64 weight) {
        SparseLabelWiseHistogramConstView::weight_const_iterator weightIterator = view.weights_cbegin();
        float64 binWeight = weightIterator[row] * weight;

        if (binWeight != 0) {
            sumOfWeights_ += binWeight;
            addToArray(statistics_, view.cbegin(row), indices.cbegin(), indices.getNumElements(), weight);
        }
    }

    void SparseLabelWiseStatisticVector::difference(const SparseLabelWiseStatisticVector& first,
                                                    const CompleteIndexVector& firstIndices,
                                                    const SparseLabelWiseStatisticVector& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        setArrayToDifference(statistics_, first.statistics_, second.statistics_, numElements_);
    }

    void SparseLabelWiseStatisticVector::difference(const SparseLabelWiseStatisticVector& first,
                                                    const PartialIndexVector& firstIndices,
                                                    const SparseLabelWiseStatisticVector& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        setArrayToDifference(statistics_, first.statistics_, second.statistics_, firstIndices.cbegin(), numElements_);
    }

}
