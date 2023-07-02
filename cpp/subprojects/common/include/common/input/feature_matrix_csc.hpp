/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_matrix_column_wise.hpp"

/**
 * Defines an interface for all feature matrices that provide column-wise access to the feature values of examples that
 * are stored in a sparse matrix in the compressed sparse column (CSC) format.
 */
class MLRLCOMMON_API ICscFeatureMatrix : virtual public IColumnWiseFeatureMatrix {
    public:

        virtual ~ICscFeatureMatrix() override {};
};

/**
 * Creates and returns a new object of the type `ICscFeatureMatrix`.
 *
 * @param numRows       The number of rows in the feature matrix
 * @param numCols       The number of columns in the feature matrix
 * @param data          A pointer to an array of type `float32`, shape `(num_non_zero_values)`, that stores all
 *                      non-zero feature values
 * @param rowIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
 *                      row-indices, the values in `data` correspond to
 * @param colIndices    A pointer to an array of type `uint32`, shape `(numCols + 1)`, that stores the indices
 *                      of the first element in `data` and `rowIndices` that corresponds to a certain column.
 *                      The index at the last position is equal to `num_non_zero_values`
 * @return              An unique pointer to an object of type `ICscFeatureMatrix` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ICscFeatureMatrix> createCscFeatureMatrix(uint32 numRows, uint32 numCols,
                                                                         const float32* data, uint32* rowIndices,
                                                                         uint32* colIndices);
