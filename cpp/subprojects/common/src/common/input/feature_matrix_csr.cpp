#include "common/input/feature_matrix_csr.hpp"
#include "common/output/prediction_matrix_dense.hpp"
#include "common/output/prediction_matrix_sparse_binary.hpp"
#include "common/output/predictor_classification.hpp"
#include "common/output/predictor_regression.hpp"
#include "common/output/predictor_probability.hpp"


CsrFeatureMatrix::CsrFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data, uint32* rowIndices,
                                   uint32* colIndices)
    : CsrConstView<const float32>(numRows, numCols, data, rowIndices, colIndices) {

}

bool CsrFeatureMatrix::isSparse() const {
    return true;
}

std::unique_ptr<DensePredictionMatrix<uint8>> CsrFeatureMatrix::predictLabels(
        const IClassificationPredictor& predictor, uint32 numLabels) const {
    return predictor.predict(*this, numLabels);
}

std::unique_ptr<BinarySparsePredictionMatrix> CsrFeatureMatrix::predictSparseLabels(
        const IClassificationPredictor& predictor, uint32 numLabels) const {
    return predictor.predictSparse(*this, numLabels);
}

std::unique_ptr<DensePredictionMatrix<float64>> CsrFeatureMatrix::predictScores(
        const IRegressionPredictor& predictor, uint32 numLabels) const {
    return predictor.predict(*this, numLabels);
}

std::unique_ptr<DensePredictionMatrix<float64>> CsrFeatureMatrix::predictProbabilities(
        const IProbabilityPredictor& predictor, uint32 numLabels) const {
    return predictor.predict(*this, numLabels);
}

std::unique_ptr<ICsrFeatureMatrix> createCsrFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data,
                                                          uint32* rowIndices, uint32* colIndices) {
    return std::make_unique<CsrFeatureMatrix>(numRows, numCols, data, rowIndices, colIndices);
}
