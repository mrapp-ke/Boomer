/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_csc.hpp"
#include "common/input/feature_matrix.hpp"


/**
 * Implements column-wise read-only access to the feature values of individual training examples that are stored in a
 * pre-allocated sparse matrix in the compressed sparse column (CSC) format.
 */
class CscFeatureMatrix final : public IFeatureMatrix {

    private:

        CscView<float32> view_;

    public:

        /**
         * @param numRows       The number of rows in the feature matrix
         * @param numCols       The number of columns in the feature matrix
         * @param data          A pointer to an array of type `float32`, shape `(num_non_zero_values)`, that stores all
         *                      non-zero feature values
         * @param rowIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
         *                      row-indices, the values in `data` correspond to
         * @param colIndices    A pointer to an array of type `uint32`, shape `(numCols + 1)`, that stores the indices
         *                      of the first element in `data` and `rowIndices` that corresponds to a certain column.
         *                      The index at the last position is equal to `num_non_zero_values`
         */
        CscFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data, const uint32* rowIndices,
                         const uint32* colIndices);

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        void fetchFeatureVector(uint32 featureIndex, std::unique_ptr<FeatureVector>& featureVectorPtr) const override;

};
