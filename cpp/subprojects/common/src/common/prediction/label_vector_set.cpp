#include "common/prediction/label_vector_set.hpp"

#include "common/input/feature_matrix_c_contiguous.hpp"
#include "common/input/feature_matrix_csr.hpp"
#include "common/model/rule_list.hpp"
#include "common/prediction/predictor_binary.hpp"
#include "common/prediction/predictor_probability.hpp"
#include "common/prediction/predictor_score.hpp"
#include "common/prediction/probability_calibration_joint.hpp"
#include "common/prediction/probability_calibration_marginal.hpp"

#include <unordered_map>

/**
 * Allows to compute hashes for objects of type `LabelVector`.
 */
struct LabelVectorHash final {
    public:

        /**
         * Computes and returns a hash for an object of type `LabelVector`.
         *
         * @param v A reference to an object of type `LabelVector`
         * @return  The hash that has been computed
         */
        inline std::size_t operator()(const LabelVector& v) const {
            return hashArray(v.cbegin(), v.getNumElements());
        }
};

/**
 * Allows to check whether two objects of type `LabelVector` are equal or not.
 */
struct LabelVectorPred final {
    public:

        /**
         * Returns whether two objects of type `LabelVector` are equal or not.
         *
         * @param lhs   A reference to the first object of type `LabelVector`
         * @param rhs   A reference to the second object of type `LabelVector`
         * @return      True, if the given objects are equal, false otherwise
         */
        inline bool operator()(const LabelVector& lhs, const LabelVector& rhs) const {
            return compareArrays(lhs.cbegin(), lhs.getNumElements(), rhs.cbegin(), rhs.getNumElements());
        }
};

LabelVectorSet::LabelVectorSet() {}

LabelVectorSet::LabelVectorSet(const IRowWiseLabelMatrix& labelMatrix) {
    std::unordered_map<std::reference_wrapper<LabelVector>, uint32, LabelVectorHash, LabelVectorPred> map;
    uint32 numRows = labelMatrix.getNumRows();

    for (uint32 i = 0; i < numRows; i++) {
        std::unique_ptr<LabelVector> labelVectorPtr = labelMatrix.createLabelVector(i);
        auto it = map.find(*labelVectorPtr);

        if (it == map.end()) {
            map.emplace(*labelVectorPtr, (uint32) frequencies_.size());
            frequencies_.emplace_back(1);
            labelVectors_.push_back(std::move(labelVectorPtr));
        } else {
            uint32 index = (*it).second;
            frequencies_[index] += 1;
        }
    }
}

LabelVectorSet::const_iterator LabelVectorSet::cbegin() const {
    return labelVectors_.cbegin();
}

LabelVectorSet::const_iterator LabelVectorSet::cend() const {
    return labelVectors_.cend();
}

LabelVectorSet::frequency_const_iterator LabelVectorSet::frequencies_cbegin() const {
    return frequencies_.cbegin();
}

LabelVectorSet::frequency_const_iterator LabelVectorSet::frequencies_cend() const {
    return frequencies_.cend();
}

uint32 LabelVectorSet::getNumLabelVectors() const {
    return (uint32) labelVectors_.size();
}

void LabelVectorSet::addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr, uint32 frequency) {
    labelVectors_.push_back(std::move(labelVectorPtr));
    frequencies_.emplace_back(frequency);
}

void LabelVectorSet::visit(LabelVectorVisitor visitor) const {
    uint32 numLabelVectors = this->getNumLabelVectors();

    for (uint32 i = 0; i < numLabelVectors; i++) {
        const std::unique_ptr<LabelVector>& labelVectorPtr = labelVectors_[i];
        uint32 frequency = frequencies_[i];
        visitor(*labelVectorPtr, frequency);
    }
}

std::unique_ptr<IJointProbabilityCalibrator> LabelVectorSet::createJointProbabilityCalibrator(
  const IJointProbabilityCalibratorFactory& factory,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const {
    return factory.create(marginalProbabilityCalibrationModel, this);
}

std::unique_ptr<IBinaryPredictor> LabelVectorSet::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel,
                          jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IBinaryPredictor> LabelVectorSet::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel,
                          jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ISparseBinaryPredictor> LabelVectorSet::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel,
                          jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ISparseBinaryPredictor> LabelVectorSet::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel,
                          jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IScorePredictor> LabelVectorSet::createScorePredictor(const IScorePredictorFactory& factory,
                                                                      const CContiguousFeatureMatrix& featureMatrix,
                                                                      const RuleList& model, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, numLabels);
}

std::unique_ptr<IScorePredictor> LabelVectorSet::createScorePredictor(const IScorePredictorFactory& factory,
                                                                      const CsrFeatureMatrix& featureMatrix,
                                                                      const RuleList& model, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, numLabels);
}

std::unique_ptr<IProbabilityPredictor> LabelVectorSet::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel,
                          jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IProbabilityPredictor> LabelVectorSet::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel,
                          jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ILabelVectorSet> createLabelVectorSet() {
    return std::make_unique<LabelVectorSet>();
}
