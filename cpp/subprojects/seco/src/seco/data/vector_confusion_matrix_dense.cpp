#include "seco/data/vector_confusion_matrix_dense.hpp"
#include "common/data/arrays.hpp"
#include <cstdlib>


namespace seco {

    template<typename LabelMatrix>
    static inline void addInternally(ConfusionMatrix* confusionMatrices, uint32 numElements, uint32 exampleIndex,
                                     const LabelMatrix& labelMatrix, const BinarySparseArrayVector& majorityLabelVector,
                                     const DenseWeightMatrix& weightMatrix, float64 weight) {
        auto majorityIterator = make_binary_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                             majorityLabelVector.indices_cend());
        typename DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(exampleIndex);
        typename LabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);

        for (uint32 i = 0; i < numElements; i++) {
            float64 labelWeight = weightIterator[i];

            if (labelWeight > 0) {
                bool trueLabel = *labelIterator;
                bool majorityLabel = *majorityIterator;
                ConfusionMatrix& confusionMatrix = confusionMatrices[i];
                float64& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element += (labelWeight * weight);
            }

            labelIterator++;
            majorityIterator++;
        }
    }

    template<typename LabelMatrix>
    static inline void addToSubsetInternally(ConfusionMatrix* confusionMatrices, uint32 numElements,
                                             uint32 exampleIndex, const LabelMatrix& labelMatrix,
                                             const BinarySparseArrayVector& majorityLabelVector,
                                             const DenseWeightMatrix& weightMatrix, float64 weight) {
        auto majorityIterator = make_binary_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                             majorityLabelVector.indices_cend());
        typename DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(exampleIndex);
        typename LabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);

        for (uint32 i = 0; i < numElements; i++) {
            float64 labelWeight = weightIterator[i];

            if (labelWeight > 0) {
                bool trueLabel = *labelIterator;
                bool majorityLabel = *majorityIterator;
                ConfusionMatrix& confusionMatrix = confusionMatrices[i];
                float64& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element += (labelWeight * weight);
            }

            labelIterator++;
            majorityIterator++;
        }
    }

    DenseConfusionMatrixVector::DenseConfusionMatrixVector(uint32 numElements)
        : DenseConfusionMatrixVector(numElements, false) {

    }

    DenseConfusionMatrixVector::DenseConfusionMatrixVector(uint32 numElements, bool init)
        : array_(init ? (ConfusionMatrix*) calloc(numElements, sizeof(ConfusionMatrix))
                      : (ConfusionMatrix*) malloc(numElements * sizeof(ConfusionMatrix))),
          numElements_(numElements) {

    }

    DenseConfusionMatrixVector::DenseConfusionMatrixVector(const DenseConfusionMatrixVector& other)
        : DenseConfusionMatrixVector(other.numElements_) {
        copyArray(other.array_, array_, numElements_);
    }

    DenseConfusionMatrixVector::~DenseConfusionMatrixVector() {
        free(array_);
    }

    DenseConfusionMatrixVector::iterator DenseConfusionMatrixVector::begin() {
        return array_;
    }

    DenseConfusionMatrixVector::iterator DenseConfusionMatrixVector::end() {
        return &array_[numElements_];
    }

    DenseConfusionMatrixVector::const_iterator DenseConfusionMatrixVector::cbegin() const {
        return array_;
    }

    DenseConfusionMatrixVector::const_iterator DenseConfusionMatrixVector::cend() const {
        return &array_[numElements_];
    }

    uint32 DenseConfusionMatrixVector::getNumElements() const {
        return numElements_;
    }

    void DenseConfusionMatrixVector::clear() {
        setArrayToZeros(array_, numElements_);
    }

    void DenseConfusionMatrixVector::add(const_iterator begin, const_iterator end) {
        for (uint32 i = 0; i < numElements_; i++) {
            array_[i] += begin[i];
        }
    }

    void DenseConfusionMatrixVector::add(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                         const BinarySparseArrayVector& majorityLabelVector,
                                         const DenseWeightMatrix& weightMatrix, float64 weight) {
        addInternally<CContiguousLabelMatrix>(array_, numElements_, exampleIndex, labelMatrix, majorityLabelVector,
                                              weightMatrix, weight);
    }

    void DenseConfusionMatrixVector::add(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                         const BinarySparseArrayVector& majorityLabelVector,
                                         const DenseWeightMatrix& weightMatrix, float64 weight) {
        addInternally<CsrLabelMatrix>(array_, numElements_, exampleIndex, labelMatrix, majorityLabelVector,
                                      weightMatrix, weight);
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                 const BinarySparseArrayVector& majorityLabelVector,
                                                 const DenseWeightMatrix& weightMatrix,
                                                 const CompleteIndexVector& indices, float64 weight) {
        addToSubsetInternally<CContiguousLabelMatrix>(array_, numElements_, exampleIndex, labelMatrix,
                                                      majorityLabelVector, weightMatrix, weight);
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                 const BinarySparseArrayVector& majorityLabelVector,
                                                 const DenseWeightMatrix& weightMatrix,
                                                 const CompleteIndexVector& indices, float64 weight) {
        addToSubsetInternally<CsrLabelMatrix>(array_, numElements_, exampleIndex, labelMatrix, majorityLabelVector,
                                              weightMatrix, weight);
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                 const BinarySparseArrayVector& majorityLabelVector,
                                                 const DenseWeightMatrix& weightMatrix,
                                                 const PartialIndexVector& indices, float64 weight) {
        auto majorityIterator = make_binary_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                             majorityLabelVector.indices_cend());
        typename DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(exampleIndex);
        CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        uint32 numElements = indices.getNumElements();
        uint32 previousIndex = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            float64 labelWeight = weightIterator[index];

            if (labelWeight > 0) {
                bool trueLabel = labelIterator[index];
                std::advance(majorityIterator, index - previousIndex);
                bool majorityLabel = *majorityIterator;
                ConfusionMatrix& confusionMatrix = array_[i];
                float64& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element += (labelWeight * weight);
                previousIndex = index;
            }
        }
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                 const BinarySparseArrayVector& majorityLabelVector,
                                                 const DenseWeightMatrix& weightMatrix,
                                                 const PartialIndexVector& indices, float64 weight) {
        auto majorityIterator = make_binary_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                             majorityLabelVector.indices_cend());
        typename DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(exampleIndex);
        CsrLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        uint32 numElements = indices.getNumElements();
        uint32 previousIndex = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            float64 labelWeight = weightIterator[index];

            if (labelWeight > 0) {
                std::advance(labelIterator, index - previousIndex);
                bool trueLabel = *labelIterator;
                std::advance(majorityIterator, index - previousIndex);
                bool majorityLabel = *majorityIterator;
                ConfusionMatrix& confusionMatrix = array_[i];
                float64& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element += (labelWeight * weight);
                previousIndex = index;
            }
        }
    }

    void DenseConfusionMatrixVector::difference(const_iterator firstBegin, const_iterator firstEnd,
                                                const CompleteIndexVector& firstIndices, const_iterator secondBegin,
                                                const_iterator secondEnd) {
        setArrayToDifference(array_, firstBegin, secondBegin, numElements_);
    }

    void DenseConfusionMatrixVector::difference(const_iterator firstBegin, const_iterator firstEnd,
                                                const PartialIndexVector& firstIndices, const_iterator secondBegin,
                                                const_iterator secondEnd) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        setArrayToDifference(array_, firstBegin, secondBegin, indexIterator, numElements_);
    }

}
