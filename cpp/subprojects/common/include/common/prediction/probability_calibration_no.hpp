/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/probability_calibration_joint.hpp"

/**
 * Defines an interface for all models for the calibration of marginal or joint probabilities that do make any
 * adjustments.
 */
class MLRLCOMMON_API INoProbabilityCalibrationModel : public IMarginalProbabilityCalibrationModel,
                                                      public IJointProbabilityCalibrationModel {
    public:

        virtual ~INoProbabilityCalibrationModel() override {};
};

/**
 * A factory that allows to create instances of the type `IMarginalProbabilityCalibrator` that do not fit a model for
 * the calibration of marginal probabilities.
 */
class NoMarginalProbabilityCalibratorFactory final : public IMarginalProbabilityCalibratorFactory {
    public:

        virtual ~NoMarginalProbabilityCalibratorFactory() {};

        std::unique_ptr<IMarginalProbabilityCalibrator> create() const override;
};

/**
 * Allows to configure a calibrator that does not fit a model for the calibration of marginal probabilities.
 */
class NoMarginalProbabilityCalibratorConfig final : public IMarginalProbabilityCalibratorConfig {
    public:

        bool shouldUseHoldoutSet() const override;

        std::unique_ptr<IMarginalProbabilityCalibratorFactory> createMarginalProbabilityCalibratorFactory()
          const override;
};

/**
 * A factory that allows to create instances of the class `IJointProbabilityCalibrator` that do not fit a model for the
 * calibration of joint probabilities.
 */
class NoJointProbabilityCalibratorFactory final : public IJointProbabilityCalibratorFactory {
    public:

        std::unique_ptr<IJointProbabilityCalibrator> create(
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const LabelVectorSet* labelVectorSet) const override;
};

/**
 * Allows to configure a calibrator that does not fit a model for the calibration of joint probabilities.
 */
class NoJointProbabilityCalibratorConfig final : public IJointProbabilityCalibratorConfig {
    public:

        bool shouldUseHoldoutSet() const override;

        bool isLabelVectorSetNeeded() const override;

        std::unique_ptr<IJointProbabilityCalibratorFactory> createJointProbabilityCalibratorFactory() const override;
};

/**
 * Creates and returns a new object of the type `INoProbabilityCalibrationModel`.
 *
 * @return An unique pointer to an object of type `INoProbabilityCalibrationModel` that has been created
 */
MLRLCOMMON_API std::unique_ptr<INoProbabilityCalibrationModel> createNoProbabilityCalibrationModel();
