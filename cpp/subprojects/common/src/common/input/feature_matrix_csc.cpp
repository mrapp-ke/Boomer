#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "common/input/feature_matrix_csc.hpp"

#include "common/data/view_csc.hpp"

/**
 * An implementation of the type `ICscFeatureMatrix` that provides column-wise read-only access to the feature values of
 * examples that are stored in a pre-allocated sparse matrix in the compressed sparse column (CSC) format.
 */
class CscFeatureMatrix final : public CscConstView<const float32>,
                               virtual public ICscFeatureMatrix {
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
        CscFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data, uint32* rowIndices, uint32* colIndices)
            : CscConstView<const float32>(numRows, numCols, data, rowIndices, colIndices) {}

        bool isSparse() const override {
            return true;
        }

        void fetchFeatureVector(uint32 featureIndex, std::unique_ptr<FeatureVector>& featureVectorPtr) const override {
            CscConstView<const float32>::index_const_iterator indexIterator = this->indices_cbegin(featureIndex);
            CscConstView<const float32>::index_const_iterator indicesEnd = this->indices_cend(featureIndex);
            CscConstView<const float32>::value_const_iterator valueIterator = this->values_cbegin(featureIndex);
            uint32 numElements = indicesEnd - indexIterator;
            featureVectorPtr = std::make_unique<FeatureVector>(numElements);
            FeatureVector::iterator vectorIterator = featureVectorPtr->begin();
            uint32 i = 0;

            for (uint32 j = 0; j < numElements; j++) {
                uint32 index = indexIterator[j];
                float32 value = valueIterator[j];

                if (std::isnan(value)) {
                    featureVectorPtr->addMissingIndex(index);
                } else {
                    vectorIterator[i].index = index;
                    vectorIterator[i].value = value;
                    i++;
                }
            }

            featureVectorPtr->setNumElements(i, true);
        }
};

std::unique_ptr<ICscFeatureMatrix> createCscFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data,
                                                          uint32* rowIndices, uint32* colIndices) {
    return std::make_unique<CscFeatureMatrix>(numRows, numCols, data, rowIndices, colIndices);
}

#ifdef _WIN32
    #pragma warning(pop)
#endif
