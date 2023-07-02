/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "common/data/view_csr.hpp"
#include "common/input/feature_matrix_row_wise.hpp"

/**
 * Defines an interface for all feature matrices that provide row-wise access to the feature values of examples that are
 * stored in a sparse matrix in the compressed sparse row (CSR) format.
 */
class MLRLCOMMON_API ICsrFeatureMatrix : virtual public IRowWiseFeatureMatrix {
    public:

        virtual ~ICsrFeatureMatrix() override {};
};

/**
 * An implementation of the type `ICsrFeatureMatrix` that provides row-wise read-only access to the feature values of
 * examples that are stored in a sparse matrix in the compressed sparse row (CSR) format.
 */
class CsrFeatureMatrix final : public CsrConstView<const float32>,
                               virtual public ICsrFeatureMatrix {
    public:

        /**
         * @param numRows       The number of rows in the feature matrix
         * @param numCols       The number of columns in the feature matrix
         * @param data          A pointer to an array of type `float32`, shape `(num_non_zero_values)`, that stores all
         *                      non-zero values
         * @param rowIndices    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices
         *                      of the first element in `data` and `colIndices` that corresponds to a certain row. The
         *                      index at the last position is equal to `num_non_zero_values`
         * @param colIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
         *                      column-indices, the values in `data` correspond to
         */
        CsrFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data, uint32* rowIndices, uint32* colIndices);

        bool isSparse() const override;

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const IRuleModel& ruleModel,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const IRuleModel& ruleModel,
                                                              const ILabelSpaceInfo& labelSpaceInfo,
                                                              uint32 numLabels) const override;

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const IRuleModel& ruleModel,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;
};

/**
 * Creates and returns a new object of the type `ICsrFeatureMatrix`.
 *
 * @param numRows       The number of rows in the feature matrix
 * @param numCols       The number of columns in the feature matrix
 * @param data          A pointer to an array of type `float32`, shape `(num_non_zero_values)`, that stores all non-zero
 *                      values
 * @param rowIndices    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices of the
 *                      first element in `data` and `colIndices` that corresponds to a certain row. The index at the
 *                      last position is equal to `num_non_zero_values`
 * @param colIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
 *                      column-indices, the values in `data` correspond to
 * @return              An unique pointer to an object of type `ICsrFeatureMatrix` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ICsrFeatureMatrix> createCsrFeatureMatrix(uint32 numRows, uint32 numCols,
                                                                         const float32* data, uint32* rowIndices,
                                                                         uint32* colIndices);

#ifdef _WIN32
    #pragma warning(pop)
#endif
