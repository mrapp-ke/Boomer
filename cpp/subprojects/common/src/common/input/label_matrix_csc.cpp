#include "common/input/label_matrix_csc.hpp"
#include "common/data/arrays.hpp"
#include <cstdlib>


template<typename IndexIterator>
static inline uint32* copyLabelMatrix(uint32* rowIndices, uint32* colIndices, const CContiguousLabelMatrix& labelMatrix,
                                      IndexIterator indicesBegin, IndexIterator indicesEnd) {
    uint32 numExamples = indicesEnd - indicesBegin;
    uint32 numLabels = labelMatrix.getNumCols();
    uint32 n = 0;

    for (uint32 i = 0; i < numLabels; i++) {
        colIndices[i] = n;

        for (uint32 j = 0; j < numExamples; j++) {
            uint32 exampleIndex = indicesBegin[j];

            if (labelMatrix.row_values_cbegin(exampleIndex)[i]) {
                rowIndices[n] = exampleIndex;
                n++;
            }
        }
    }

    colIndices[numLabels] = n;
    return (uint32*) realloc(rowIndices, n * sizeof(uint32));
}

template<typename IndexIterator>
static inline uint32* copyLabelMatrix(uint32* rowIndices, uint32* colIndices, const CsrLabelMatrix& labelMatrix,
                                      IndexIterator indicesBegin, IndexIterator indicesEnd) {
    uint32 numExamples = indicesEnd - indicesBegin;
    uint32 numLabels = labelMatrix.getNumCols();

    // Set column indices of the CSC matrix to zero...
    setArrayToZeros(colIndices, numLabels);

    // Determine the number of non-zero elements per column...
    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indicesBegin[i];
        CsrLabelMatrix::index_const_iterator labelIndexIterator = labelMatrix.row_indices_cbegin(exampleIndex);
        uint32 numRelevantLabels = labelMatrix.row_indices_cend(exampleIndex) - labelIndexIterator;

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
        CsrLabelMatrix::index_const_iterator labelIndexIterator = labelMatrix.row_indices_cbegin(exampleIndex);
        uint32 numRelevantLabels = labelMatrix.row_indices_cend(exampleIndex) - labelIndexIterator;

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

CscLabelMatrix::CscLabelMatrix(const CContiguousLabelMatrix& labelMatrix,
                               CompleteIndexVector::const_iterator indicesBegin,
                               CompleteIndexVector::const_iterator indicesEnd)
    : rowIndices_((uint32*) malloc(labelMatrix.getNumRows() * labelMatrix.getNumCols() * sizeof(uint32))),
      colIndices_((uint32*) malloc((labelMatrix.getNumCols() + 1) * sizeof(uint32))),
      view_(BinaryCscConstView(labelMatrix.getNumRows(), labelMatrix.getNumCols(), rowIndices_, colIndices_)) {
    rowIndices_ = copyLabelMatrix<CompleteIndexVector::const_iterator>(rowIndices_, colIndices_, labelMatrix,
                                                                       indicesBegin, indicesEnd);
}

CscLabelMatrix::CscLabelMatrix(const CContiguousLabelMatrix& labelMatrix,
                               PartialIndexVector::const_iterator indicesBegin,
                               PartialIndexVector::const_iterator indicesEnd)
    : rowIndices_((uint32*) malloc(labelMatrix.getNumRows() * labelMatrix.getNumCols() * sizeof(uint32))),
      colIndices_((uint32*) malloc((labelMatrix.getNumCols() + 1) * sizeof(uint32))),
      view_(BinaryCscConstView(labelMatrix.getNumRows(), labelMatrix.getNumCols(), rowIndices_, colIndices_)) {
    rowIndices_ = copyLabelMatrix<PartialIndexVector::const_iterator>(rowIndices_, colIndices_, labelMatrix,
                                                                      indicesBegin, indicesEnd);
}

CscLabelMatrix::CscLabelMatrix(const CsrLabelMatrix& labelMatrix, CompleteIndexVector::const_iterator indicesBegin,
                               CompleteIndexVector::const_iterator indicesEnd)
    : rowIndices_((uint32*) malloc(labelMatrix.getNumNonZeroElements() * sizeof(uint32))),
      colIndices_((uint32*) malloc((labelMatrix.getNumCols() + 1) * sizeof(uint32))),
      view_(BinaryCscConstView(labelMatrix.getNumRows(), labelMatrix.getNumCols(), rowIndices_, colIndices_)) {
    rowIndices_ = copyLabelMatrix<CompleteIndexVector::const_iterator>(rowIndices_, colIndices_, labelMatrix,
                                                                       indicesBegin, indicesEnd);
}

CscLabelMatrix::CscLabelMatrix(const CsrLabelMatrix& labelMatrix, PartialIndexVector::const_iterator indicesBegin,
                               PartialIndexVector::const_iterator indicesEnd)
    : rowIndices_((uint32*) malloc(labelMatrix.getNumNonZeroElements() * sizeof(uint32))),
      colIndices_((uint32*) malloc((labelMatrix.getNumCols() + 1) * sizeof(uint32))),
      view_(BinaryCscConstView(labelMatrix.getNumRows(), labelMatrix.getNumCols(), rowIndices_, colIndices_)) {
    rowIndices_ = copyLabelMatrix<PartialIndexVector::const_iterator>(rowIndices_, colIndices_, labelMatrix,
                                                                      indicesBegin, indicesEnd);
}

CscLabelMatrix::~CscLabelMatrix() {
    free(rowIndices_);
    free(colIndices_);
}

CscLabelMatrix::index_const_iterator CscLabelMatrix::column_indices_cbegin(uint32 col) const {
    return view_.column_indices_cbegin(col);
}

CscLabelMatrix::index_const_iterator CscLabelMatrix::column_indices_cend(uint32 col) const {
    return view_.column_indices_cend(col);
}

CscLabelMatrix::value_const_iterator CscLabelMatrix::column_values_cbegin(uint32 col) const {
    return view_.column_values_cbegin(col);
}

CscLabelMatrix::value_const_iterator CscLabelMatrix::column_values_cend(uint32 col) const {
    return view_.column_values_cend(col);
}

uint32 CscLabelMatrix::getNumRows() const {
    return view_.getNumRows();
}

uint32 CscLabelMatrix::getNumCols() const {
    return view_.getNumCols();
}

uint32 CscLabelMatrix::getNumNonZeroElements() const {
    return view_.getNumNonZeroElements();
}
