#include "common/input/label_matrix_csc.hpp"

#include "common/data/arrays.hpp"

#include <cstdlib>

template<typename IndexIterator>
static inline uint32* copyLabelMatrix(uint32* rowIndices, uint32* colIndices,
                                      const CContiguousConstView<const uint8>& labelMatrix, IndexIterator indicesBegin,
                                      IndexIterator indicesEnd) {
    uint32 numExamples = indicesEnd - indicesBegin;
    uint32 numLabels = labelMatrix.getNumCols();
    uint32 n = 0;

    for (uint32 i = 0; i < numLabels; i++) {
        colIndices[i] = n;

        for (uint32 j = 0; j < numExamples; j++) {
            uint32 exampleIndex = indicesBegin[j];

            if (labelMatrix.values_cbegin(exampleIndex)[i]) {
                rowIndices[n] = exampleIndex;
                n++;
            }
        }
    }

    colIndices[numLabels] = n;
    return (uint32*) realloc(rowIndices, n * sizeof(uint32));
}

template<typename IndexIterator>
static inline uint32* copyLabelMatrix(uint32* rowIndices, uint32* colIndices, const BinaryCsrConstView& labelMatrix,
                                      IndexIterator indicesBegin, IndexIterator indicesEnd) {
    uint32 numExamples = indicesEnd - indicesBegin;
    uint32 numLabels = labelMatrix.getNumCols();

    // Set column indices of the CSC matrix to zero...
    setArrayToZeros(colIndices, numLabels);

    // Determine the number of non-zero elements per column...
    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indicesBegin[i];
        BinaryCsrConstView::index_const_iterator labelIndexIterator = labelMatrix.indices_cbegin(exampleIndex);
        uint32 numRelevantLabels = labelMatrix.indices_cend(exampleIndex) - labelIndexIterator;

        for (uint32 j = 0; j < numRelevantLabels; j++) {
            uint32 labelIndex = labelIndexIterator[j];
            colIndices[labelIndex]++;
        }
    }

    // Update the column indices of the CSC matrix with respect to the number of non-zero elements that correspond to
    // previous columns...
    uint32 tmp = 0;

    for (uint32 i = 0; i < numLabels; i++) {
        uint32 labelIndex = colIndices[i];
        colIndices[i] = tmp;
        tmp += labelIndex;
    }

    // Set the row indices of the CSC matrix. This will modify the column indices...
    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indicesBegin[i];
        BinaryCsrConstView::index_const_iterator labelIndexIterator = labelMatrix.indices_cbegin(exampleIndex);
        uint32 numRelevantLabels = labelMatrix.indices_cend(exampleIndex) - labelIndexIterator;

        for (uint32 j = 0; j < numRelevantLabels; j++) {
            uint32 originalLabelIndex = labelIndexIterator[j];
            uint32 labelIndex = colIndices[originalLabelIndex];
            rowIndices[labelIndex] = exampleIndex;
            colIndices[originalLabelIndex]++;
        }
    }

    // Reset the column indices to the previous values...
    tmp = 0;

    for (uint32 i = 0; i < numLabels; i++) {
        uint32 labelIndex = colIndices[i];
        colIndices[i] = tmp;
        tmp = labelIndex;
    }

    colIndices[numLabels] = tmp;
    return (uint32*) realloc(rowIndices, tmp * sizeof(uint32));
}

CscLabelMatrix::CscLabelMatrix(const CContiguousConstView<const uint8>& labelMatrix,
                               CompleteIndexVector::const_iterator indicesBegin,
                               CompleteIndexVector::const_iterator indicesEnd)
    : BinaryCscConstView(labelMatrix.getNumRows(), labelMatrix.getNumCols(),
                         (uint32*) malloc(labelMatrix.getNumRows() * labelMatrix.getNumCols() * sizeof(uint32)),
                         (uint32*) malloc((labelMatrix.getNumCols() + 1) * sizeof(uint32))) {
    this->rowIndices_ = copyLabelMatrix<CompleteIndexVector::const_iterator>(this->rowIndices_, this->colIndices_,
                                                                             labelMatrix, indicesBegin, indicesEnd);
}

CscLabelMatrix::CscLabelMatrix(const CContiguousConstView<const uint8>& labelMatrix,
                               PartialIndexVector::const_iterator indicesBegin,
                               PartialIndexVector::const_iterator indicesEnd)
    : BinaryCscConstView(labelMatrix.getNumRows(), labelMatrix.getNumCols(),
                         (uint32*) malloc(labelMatrix.getNumRows() * labelMatrix.getNumCols() * sizeof(uint32)),
                         (uint32*) malloc((labelMatrix.getNumCols() + 1) * sizeof(uint32))) {
    this->rowIndices_ = copyLabelMatrix<PartialIndexVector::const_iterator>(this->rowIndices_, this->colIndices_,
                                                                            labelMatrix, indicesBegin, indicesEnd);
}

CscLabelMatrix::CscLabelMatrix(const BinaryCsrConstView& labelMatrix, CompleteIndexVector::const_iterator indicesBegin,
                               CompleteIndexVector::const_iterator indicesEnd)
    : BinaryCscConstView(labelMatrix.getNumRows(), labelMatrix.getNumCols(),
                         (uint32*) malloc(labelMatrix.getNumNonZeroElements() * sizeof(uint32)),
                         (uint32*) malloc((labelMatrix.getNumCols() + 1) * sizeof(uint32))) {
    this->rowIndices_ = copyLabelMatrix<CompleteIndexVector::const_iterator>(this->rowIndices_, this->colIndices_,
                                                                             labelMatrix, indicesBegin, indicesEnd);
}

CscLabelMatrix::CscLabelMatrix(const BinaryCsrConstView& labelMatrix, PartialIndexVector::const_iterator indicesBegin,
                               PartialIndexVector::const_iterator indicesEnd)
    : BinaryCscConstView(labelMatrix.getNumRows(), labelMatrix.getNumCols(),
                         (uint32*) malloc(labelMatrix.getNumNonZeroElements() * sizeof(uint32)),
                         (uint32*) malloc((labelMatrix.getNumCols() + 1) * sizeof(uint32))) {
    this->rowIndices_ = copyLabelMatrix<PartialIndexVector::const_iterator>(this->rowIndices_, this->colIndices_,
                                                                            labelMatrix, indicesBegin, indicesEnd);
}

CscLabelMatrix::~CscLabelMatrix() {
    free(this->rowIndices_);
    free(this->colIndices_);
}
