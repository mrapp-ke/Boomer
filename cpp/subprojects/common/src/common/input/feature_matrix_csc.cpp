#include "common/input/feature_matrix_csc.hpp"


CscFeatureMatrix::CscFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data, uint32* rowIndices,
                                   uint32* colIndices)
    : view_(CscConstView<const float32>(numRows, numCols, data, rowIndices, colIndices)) {

}

CscFeatureMatrix::value_const_iterator CscFeatureMatrix::column_values_cbegin(uint32 col) const {
    return view_.column_values_cbegin(col);
}

CscFeatureMatrix::value_const_iterator CscFeatureMatrix::column_values_cend(uint32 col) const {
    return view_.column_values_cend(col);
}

CscFeatureMatrix::index_const_iterator CscFeatureMatrix::column_indices_cbegin(uint32 col) const {
    return view_.column_indices_cbegin(col);
}

CscFeatureMatrix::index_const_iterator CscFeatureMatrix::column_indices_cend(uint32 col) const {
    return view_.column_indices_cend(col);
}

uint32 CscFeatureMatrix::getNumRows() const {
    return view_.getNumRows();
}

uint32 CscFeatureMatrix::getNumCols() const {
    return view_.getNumCols();
}

uint32 CscFeatureMatrix::getNumNonZeroElements() const {
    return view_.getNumNonZeroElements();
}

void CscFeatureMatrix::fetchFeatureVector(uint32 featureIndex, std::unique_ptr<FeatureVector>& featureVectorPtr) const {
    CscConstView<const float32>::index_const_iterator indexIterator = view_.column_indices_cbegin(featureIndex);
    CscConstView<const float32>::index_const_iterator indicesEnd = view_.column_indices_cend(featureIndex);
    CscConstView<const float32>::value_const_iterator valueIterator = view_.column_values_cbegin(featureIndex);
    uint32 numElements = indicesEnd - indexIterator;
    featureVectorPtr = std::make_unique<FeatureVector>(numElements);
    FeatureVector::iterator vectorIterator = featureVectorPtr->begin();
    uint32 i = 0;

    for (uint32 j = 0; j < numElements; j++) {
        uint32 index = indexIterator[j];
        float32 value = valueIterator[j];

        if (value != value) {
            // The value is NaN (because comparisons to NaN always evaluate to false)...
            featureVectorPtr->addMissingIndex(index);
        } else {
            vectorIterator[i].index = index;
            vectorIterator[i].value = value;
            i++;
        }
    }

    featureVectorPtr->setNumElements(i, true);
}
