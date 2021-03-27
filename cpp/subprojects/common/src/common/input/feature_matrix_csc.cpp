#include "common/input/feature_matrix_csc.hpp"


CscFeatureMatrix::CscFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data, const uint32* rowIndices,
                                   const uint32* colIndices)
    : view_(CscView<float32>(numRows, numCols, data, rowIndices, colIndices)) {

}

uint32 CscFeatureMatrix::getNumRows() const {
    return view_.getNumRows();
}

uint32 CscFeatureMatrix::getNumCols() const {
    return view_.getNumCols();
}

void CscFeatureMatrix::fetchFeatureVector(uint32 featureIndex, std::unique_ptr<FeatureVector>& featureVectorPtr) const {
    CscView<float32>::index_const_iterator indexIterator = view_.column_indices_cbegin(featureIndex);
    CscView<float32>::value_const_iterator valueIterator = view_.column_values_cbegin(featureIndex);
    uint32 numElements = view_.getNumNonZeroElements(featureIndex);
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
