#include "common/input/feature_matrix_fortran_contiguous.hpp"


FortranContiguousFeatureMatrix::FortranContiguousFeatureMatrix(uint32 numRows, uint32 numCols, const float32* array)
    : view_(FortranContiguousConstView<const float32>(numRows, numCols, array)) {

}

FortranContiguousFeatureMatrix::const_iterator FortranContiguousFeatureMatrix::column_cbegin(uint32 col) const {
    return view_.column_cbegin(col);
}

FortranContiguousFeatureMatrix::const_iterator FortranContiguousFeatureMatrix::column_cend(uint32 col) const {
    return view_.column_cend(col);
}

uint32 FortranContiguousFeatureMatrix::getNumRows() const {
    return view_.getNumRows();
}

uint32 FortranContiguousFeatureMatrix::getNumCols() const {
    return view_.getNumCols();
}

void FortranContiguousFeatureMatrix::fetchFeatureVector(uint32 featureIndex,
                                                        std::unique_ptr<FeatureVector>& featureVectorPtr) const {
    FortranContiguousConstView<const float32>::const_iterator columnIterator = view_.column_cbegin(featureIndex);
    uint32 numElements = this->getNumRows();
    featureVectorPtr = std::make_unique<FeatureVector>(numElements);
    FeatureVector::iterator vectorIterator = featureVectorPtr->begin();
    uint32 i = 0;

    for (uint32 j = 0; j < numElements; j++) {
        float32 value = columnIterator[j];

        if (value != value) {
            // The value is NaN (because comparisons to NaN always evaluate to false)...
            featureVectorPtr->addMissingIndex(j);
        } else {
            vectorIterator[i].index = j;
            vectorIterator[i].value = value;
            i++;
        }
    }

    featureVectorPtr->setNumElements(i, true);
}
