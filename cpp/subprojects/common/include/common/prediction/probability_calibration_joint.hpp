/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/label_vector_set.hpp"
#include "common/prediction/probability_calibration_marginal.hpp"

/**
 * Defines an interface for all classes that implement a model for the calibration of joint probabilities.
 */
class MLRLCOMMON_API IJointProbabilityCalibrationModel {
    public:

        virtual ~IJointProbabilityCalibrationModel() {};

        /**
         * Calibrates a joint probability.
         *
         * @param labelVectorIndex  The index of the label vector, the probability is predicted for
         * @param jointProbability  The joint probability to be calibrated
         * @return                  The calibrated probability
         */
        virtual float64 calibrateJointProbability(uint32 labelVectorIndex, float64 jointProbability) const = 0;
};

/**
 * Defines an interface for all classes that implement a method for fitting models for the calibration of joint
 * probabilities.
 */
class IJointProbabilityCalibrator : public IProbabilityCalibrator<IJointProbabilityCalibrationModel> {
    public:

        virtual ~IJointProbabilityCalibrator() override {};
};

/**
 * Defines an interface for all classes that allow to create instances of the type `IJointProbabilityCalibrator`.
 */
class IJointProbabilityCalibratorFactory {
    public:

        virtual ~IJointProbabilityCalibratorFactory() {};

        /**
         * Creates and returns a new object of type `IJointProbabilityCalibrator`.
         *
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param labelVectorSet                      A pointer to an object of type `LabelVectorSet` that stores all
         *                                            known label vectors or a null pointer, if no such object is
         *                                            available
         * @return                                    An unique pointer to an object of type
         *                                            `IJointProbabilityCalibrator` that has been created
         */
        virtual std::unique_ptr<IJointProbabilityCalibrator> create(
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const LabelVectorSet* labelVectorSet) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for fitting a model for the calibration of
 * joint probabilities.
 */
class IJointProbabilityCalibratorConfig {
    public:

        virtual ~IJointProbabilityCalibratorConfig() {};

        /**
         * Returns whether a holdout set should be used, if available, or not.
         *
         * @return True, if a holdout set should be used, false otherwise
         */
        virtual bool shouldUseHoldoutSet() const = 0;

        /**
         * Returns whether the calibrator needs access to the label vectors that are encountered in the training data or
         * not.
         *
         * @return True, if the calibrator needs access to the label vectors that are encountered in the training data,
         *         false otherwise
         */
        virtual bool isLabelVectorSetNeeded() const = 0;

        /**
         * Creates and returns a new object of template type `IJointProbabilityCalibratorFactory` according to the
         * configuration.
         *
         * @return An unique pointer to an object of template type `IJointProbabilityCalibratorFactory` that has been
         *         created
         */
        virtual std::unique_ptr<IJointProbabilityCalibratorFactory> createJointProbabilityCalibratorFactory() const = 0;
};
