#include "common/input/feature_matrix_c_contiguous.hpp"
#include "common/output/prediction_matrix_dense.hpp"
#include "common/output/prediction_matrix_sparse_binary.hpp"
#include "common/output/predictor_classification.hpp"
#include "common/output/predictor_regression.hpp"
#include "common/output/predictor_probability.hpp"


CContiguousFeatureMatrix::CContiguousFeatureMatrix(uint32 numRows, uint32 numCols, const float32* array)
    : CContiguousConstView<const float32>(numRows, numCols, array) {

}

bool CContiguousFeatureMatrix::isSparse() const {
    return false;
}

std::unique_ptr<DensePredictionMatrix<uint8>> CContiguousFeatureMatrix::predictLabels(
        const IClassificationPredictor& predictor, uint32 numLabels) const {
    return predictor.predict(*this, numLabels);
}

std::unique_ptr<BinarySparsePredictionMatrix> CContiguousFeatureMatrix::predictSparseLabels(
        const IClassificationPredictor& predictor, uint32 numLabels) const {
    return predictor.predictSparse(*this, numLabels);
}

std::unique_ptr<DensePredictionMatrix<float64>> CContiguousFeatureMatrix::predictScores(
        const IRegressionPredictor& predictor, uint32 numLabels) const {
    return predictor.predict(*this, numLabels);
}

std::unique_ptr<DensePredictionMatrix<float64>> CContiguousFeatureMatrix::predictProbabilities(
        const IProbabilityPredictor& predictor, uint32 numLabels) const {
    return predictor.predict(*this, numLabels);
}

std::unique_ptr<ICContiguousFeatureMatrix> createCContiguousFeatureMatrix(uint32 numRows, uint32 numCols,
                                                                          const float32* array) {
    return std::make_unique<CContiguousFeatureMatrix>(numRows, numCols, array);
}
