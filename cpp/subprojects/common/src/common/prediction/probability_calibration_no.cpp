#include "common/prediction/probability_calibration_no.hpp"

/**
 * A model for the calibration of marginal or joint probabilities that does not make any adjustments.
 */
class NoProbabilityCalibrationModel final : public INoProbabilityCalibrationModel {
    public:

        float64 calibrateMarginalProbability(uint32 labelIndex, float64 marginalProbability) const override {
            return marginalProbability;
        }

        float64 calibrateJointProbability(uint32 labelVectorIndex, float64 jointProbability) const override {
            return jointProbability;
        }
};

/**
 * An implementation of the type `IMarginalProbabilityCalibrator` that does not fit a model for the calibration of
 * marginal probabilities.
 */
class NoMarginalProbabilityCalibrator final : public IMarginalProbabilityCalibrator {
    public:

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<NoProbabilityCalibrationModel>();
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CsrLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<NoProbabilityCalibrationModel>();
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          BiPartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<NoProbabilityCalibrationModel>();
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics) const override {
            return std::make_unique<NoProbabilityCalibrationModel>();
        }
};

std::unique_ptr<IMarginalProbabilityCalibrator> NoMarginalProbabilityCalibratorFactory::create() const {
    return std::make_unique<NoMarginalProbabilityCalibrator>();
}

bool NoMarginalProbabilityCalibratorConfig::shouldUseHoldoutSet() const {
    return false;
}

std::unique_ptr<IMarginalProbabilityCalibratorFactory>
  NoMarginalProbabilityCalibratorConfig::createMarginalProbabilityCalibratorFactory() const {
    return std::make_unique<NoMarginalProbabilityCalibratorFactory>();
}

/**
 * An implementation of the type `IJointProbabilityCalibrator` that does not fit a model for the calibration of joint
 * probabilities.
 */
class NoJointProbabilityCalibrator final : public IJointProbabilityCalibrator {
    public:

        std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<NoProbabilityCalibrationModel>();
        }

        std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CsrLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<NoProbabilityCalibrationModel>();
        }

        std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          BiPartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<NoProbabilityCalibrationModel>();
        }

        std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics) const override {
            return std::make_unique<NoProbabilityCalibrationModel>();
        }
};

std::unique_ptr<IJointProbabilityCalibrator> NoJointProbabilityCalibratorFactory::create(
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const LabelVectorSet* labelVectorSet) const {
    return std::make_unique<NoJointProbabilityCalibrator>();
}

bool NoJointProbabilityCalibratorConfig::shouldUseHoldoutSet() const {
    return false;
}

bool NoJointProbabilityCalibratorConfig::isLabelVectorSetNeeded() const {
    return false;
}

std::unique_ptr<IJointProbabilityCalibratorFactory>
  NoJointProbabilityCalibratorConfig::createJointProbabilityCalibratorFactory() const {
    return std::make_unique<NoJointProbabilityCalibratorFactory>();
}

std::unique_ptr<INoProbabilityCalibrationModel> createNoProbabilityCalibrationModel() {
    return std::make_unique<NoProbabilityCalibrationModel>();
}
