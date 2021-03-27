/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_fortran_contiguous.hpp"
#include "common/input/feature_matrix.hpp"


/**
 * Implements column-wise read-only access to the feature values of individual training examples that are stored in a
 * pre-allocated Fortran-contiguous array.
 */
class FortranContiguousFeatureMatrix final : public IFeatureMatrix {

    private:

        FortranContiguousView<float32> view_;

    public:

        /**
         * @param numRows   The number of rows in the feature matrix
         * @param numCols   The number of columns in the feature matrix
         * @param array     A pointer to a Fortran-contiguous array of type `float32` that stores the feature values
         */
        FortranContiguousFeatureMatrix(uint32 numRows, uint32 numCols, float32* array);

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        void fetchFeatureVector(uint32 featureIndex, std::unique_ptr<FeatureVector>& featureVectorPtr) const override;

};
