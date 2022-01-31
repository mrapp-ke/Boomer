#include "boosting/data/statistic_vector_label_wise_dense.hpp"
#include "boosting/data/arrays.hpp"
#include "common/data/arrays.hpp"
#include <cstdlib>


namespace boosting {

    DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(uint32 numElements)
        : DenseLabelWiseStatisticVector(numElements, false) {

    }

    DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(uint32 numElements, bool init)
        : numElements_(numElements),
          statistics_((Tuple<float64>*) (init ? calloc(numElements, sizeof(Tuple<float64>))
                                              : malloc(numElements * sizeof(Tuple<float64>)))) {

    }

    DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(const DenseLabelWiseStatisticVector& vector)
        : DenseLabelWiseStatisticVector(vector.numElements_) {
        copyArray(vector.statistics_, statistics_, numElements_);
    }

    DenseLabelWiseStatisticVector::~DenseLabelWiseStatisticVector() {
        free(statistics_);
    }

    DenseLabelWiseStatisticVector::iterator DenseLabelWiseStatisticVector::begin() {
        return statistics_;
    }

    DenseLabelWiseStatisticVector::iterator DenseLabelWiseStatisticVector::end() {
        return &statistics_[numElements_];
    }

    DenseLabelWiseStatisticVector::const_iterator DenseLabelWiseStatisticVector::cbegin() const {
        return statistics_;
    }

    DenseLabelWiseStatisticVector::const_iterator DenseLabelWiseStatisticVector::cend() const {
        return &statistics_[numElements_];
    }

    uint32 DenseLabelWiseStatisticVector::getNumElements() const {
        return numElements_;
    }

    void DenseLabelWiseStatisticVector::clear() {
        setArrayToZeros(statistics_, numElements_);
    }

    void DenseLabelWiseStatisticVector::add(const_iterator begin, const_iterator end) {
        addToArray(statistics_, begin, numElements_);
    }

    void DenseLabelWiseStatisticVector::add(const_iterator begin, const_iterator end, float64 weight) {
        addToArray(statistics_, begin, numElements_, weight);
    }

    void DenseLabelWiseStatisticVector::addToSubset(const_iterator begin, const_iterator end,
                                                    const CompleteIndexVector& indices, float64 weight) {
        addToArray(statistics_, begin, numElements_, weight);
    }

    void DenseLabelWiseStatisticVector::addToSubset(const_iterator begin, const_iterator end,
                                                    const PartialIndexVector& indices, float64 weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToArray(statistics_, begin, indexIterator, numElements_, weight);
    }

    void DenseLabelWiseStatisticVector::difference(const_iterator firstBegin, const_iterator firstEnd,
                                                   const CompleteIndexVector& firstIndices, const_iterator secondBegin,
                                                   const_iterator secondEnd) {
        setArrayToDifference(statistics_, firstBegin, secondBegin, numElements_);
    }

    void DenseLabelWiseStatisticVector::difference(const_iterator firstBegin, const_iterator firstEnd,
                                                   const PartialIndexVector& firstIndices, const_iterator secondBegin,
                                                   const_iterator secondEnd) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        setArrayToDifference(statistics_, firstBegin, secondBegin, indexIterator, numElements_);
    }

}
