#include "seco/data/matrix_weight_dense.hpp"
#include "common/data/arrays.hpp"
#include "common/iterator/binary_forward_iterator.hpp"


namespace seco {

    DenseWeightMatrix::DenseWeightMatrix(uint32 numRows, uint32 numCols)
        : DenseMatrix<float64>(numRows, numCols), sumOfUncoveredWeights_(0) {
        setArrayToValue<float64>(this->array_, numRows * numCols, 1);
    }

    float64 DenseWeightMatrix::getSumOfUncoveredWeights() const {
        return sumOfUncoveredWeights_;
    }

    void DenseWeightMatrix::setSumOfUncoveredWeights(float64 sumOfUncoveredWeights) {
        sumOfUncoveredWeights_ = sumOfUncoveredWeights;
    }

    void DenseWeightMatrix::updateRow(uint32 row, const BinarySparseArrayVector& majorityLabelVector,
                                      DenseVector<float64>::const_iterator predictionBegin,
                                      DenseVector<float64>::const_iterator predictionEnd,
                                      CompleteIndexVector::const_iterator indicesBegin,
                                      CompleteIndexVector::const_iterator indicesEnd) {
        uint32 numCols = this->getNumCols();
        iterator weightIterator = this->row_begin(row);
        auto majorityIterator = make_binary_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                             majorityLabelVector.indices_cend());

        for (uint32 i = 0; i < numCols; i++) {
            bool predictedLabel = predictionBegin[i];
            bool majorityLabel = *majorityIterator;

            if (predictedLabel != majorityLabel) {
                float64 labelWeight = weightIterator[i];

                if (labelWeight > 0) {
                    sumOfUncoveredWeights_ -= labelWeight;
                    weightIterator[i] = 0;
                }
            }

            majorityIterator++;
        }
    }

    void DenseWeightMatrix::updateRow(uint32 row, const BinarySparseArrayVector& majorityLabelVector,
                                      DenseVector<float64>::const_iterator predictionBegin,
                                      DenseVector<float64>::const_iterator predictionEnd,
                                      PartialIndexVector::const_iterator indicesBegin,
                                      PartialIndexVector::const_iterator indicesEnd) {
        uint32 numPredictions = indicesEnd - indicesBegin;
        iterator weightIterator = this->row_begin(row);
        auto majorityIterator = make_binary_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                             majorityLabelVector.indices_cend());
        uint32 previousIndex = 0;

        for (uint32 i = 0; i < numPredictions; i++) {
            uint32 index = indicesBegin[i];
            bool predictedLabel = predictionBegin[i];
            std::advance(majorityIterator, index - previousIndex);
            bool majorityLabel = *majorityIterator;

            if (predictedLabel != majorityLabel) {
                float64 labelWeight = weightIterator[index];

                if (labelWeight > 0) {
                    sumOfUncoveredWeights_ -= labelWeight;
                    weightIterator[index] = 0;
                }
            }

            previousIndex = index;
        }
    }

}
