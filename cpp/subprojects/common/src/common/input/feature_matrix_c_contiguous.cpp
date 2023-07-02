#include "common/input/feature_matrix_c_contiguous.hpp"

#include "common/prediction/predictor_binary.hpp"
#include "common/prediction/predictor_probability.hpp"
#include "common/prediction/predictor_score.hpp"

CContiguousFeatureMatrix::CContiguousFeatureMatrix(uint32 numRows, uint32 numCols, const float32* array)
    : CContiguousConstView<const float32>(numRows, numCols, array) {}

bool CContiguousFeatureMatrix::isSparse() const {
    return false;
}

std::unique_ptr<IBinaryPredictor> CContiguousFeatureMatrix::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return ruleModel.createBinaryPredictor(factory, *this, labelSpaceInfo, marginalProbabilityCalibrationModel,
                                           jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ISparseBinaryPredictor> CContiguousFeatureMatrix::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return ruleModel.createSparseBinaryPredictor(factory, *this, labelSpaceInfo, marginalProbabilityCalibrationModel,
                                                 jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IScorePredictor> CContiguousFeatureMatrix::createScorePredictor(const IScorePredictorFactory& factory,
                                                                                const IRuleModel& ruleModel,
                                                                                const ILabelSpaceInfo& labelSpaceInfo,
                                                                                uint32 numLabels) const {
    return ruleModel.createScorePredictor(factory, *this, labelSpaceInfo, numLabels);
}

std::unique_ptr<IProbabilityPredictor> CContiguousFeatureMatrix::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return ruleModel.createProbabilityPredictor(factory, *this, labelSpaceInfo, marginalProbabilityCalibrationModel,
                                                jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ICContiguousFeatureMatrix> createCContiguousFeatureMatrix(uint32 numRows, uint32 numCols,
                                                                          const float32* array) {
    return std::make_unique<CContiguousFeatureMatrix>(numRows, numCols, array);
}
