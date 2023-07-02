#include "common/prediction/label_space_info_no.hpp"

#include "common/input/feature_matrix_c_contiguous.hpp"
#include "common/input/feature_matrix_csr.hpp"
#include "common/model/rule_list.hpp"
#include "common/prediction/predictor_binary.hpp"
#include "common/prediction/predictor_probability.hpp"
#include "common/prediction/predictor_score.hpp"
#include "common/prediction/probability_calibration_joint.hpp"

/**
 * An implementation of the type `INoLabelSpaceInfo` that does not provide any information about the label space.
 */
class NoLabelSpaceInfo final : public INoLabelSpaceInfo {
    public:

        std::unique_ptr<IJointProbabilityCalibrator> createJointProbabilityCalibrator(
          const IJointProbabilityCalibratorFactory& factory,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const override {
            return factory.create(marginalProbabilityCalibrationModel, nullptr);
        }

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix, const RuleList& model,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, marginalProbabilityCalibrationModel,
                                  jointProbabilityCalibrationModel, numLabels);
        }

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, marginalProbabilityCalibrationModel,
                                  jointProbabilityCalibrationModel, numLabels);
        }

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, marginalProbabilityCalibrationModel,
                                  jointProbabilityCalibrationModel, numLabels);
        }

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, marginalProbabilityCalibrationModel,
                                  jointProbabilityCalibrationModel, numLabels);
        }

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CContiguousFeatureMatrix& featureMatrix,
                                                              const RuleList& model, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, numLabels);
        }

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CsrFeatureMatrix& featureMatrix,
                                                              const RuleList& model, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, numLabels);
        }

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, marginalProbabilityCalibrationModel,
                                  jointProbabilityCalibrationModel, numLabels);
        }

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, marginalProbabilityCalibrationModel,
                                  jointProbabilityCalibrationModel, numLabels);
        }
};

std::unique_ptr<INoLabelSpaceInfo> createNoLabelSpaceInfo() {
    return std::make_unique<NoLabelSpaceInfo>();
}
