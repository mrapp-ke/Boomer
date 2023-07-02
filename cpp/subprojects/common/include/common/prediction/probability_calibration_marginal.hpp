/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/label_matrix_c_contiguous.hpp"
#include "common/input/label_matrix_csr.hpp"
#include "common/macros.hpp"
#include "common/prediction/probability_calibration.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/statistics/statistics.hpp"

/**
 * Defines an interface for all classes that implement a model for the calibration of marginal probabilities.
 */
class MLRLCOMMON_API IMarginalProbabilityCalibrationModel {
    public:

        virtual ~IMarginalProbabilityCalibrationModel() {};

        /**
         * Calibrates the marginal probability that is predicted for a specific label.
         *
         * @param labelIndex            The index of the label, the probability is predicted for
         * @param marginalProbability   The marginal probability to be calibrated
         * @return                      The calibrated probability
         */
        virtual float64 calibrateMarginalProbability(uint32 labelIndex, float64 marginalProbability) const = 0;
};

/**
 * Defines an interface for all classes that implement a method for fitting models for the calibration of marginal
 * probabilities.
 */
class IMarginalProbabilityCalibrator : public IProbabilityCalibrator<IMarginalProbabilityCalibrationModel> {
    public:

        virtual ~IMarginalProbabilityCalibrator() override {};
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IMarginalProbabilityCalibrator`.
 */
class IMarginalProbabilityCalibratorFactory {
    public:

        virtual ~IMarginalProbabilityCalibratorFactory() {};

        /**
         * Creates and returns a new object of type `IMarginalProbabilityCalibrator`.
         *
         * @return An unique pointer to an object of type `IMarginalProbabilityCalibrator` that has been created
         */
        virtual std::unique_ptr<IMarginalProbabilityCalibrator> create() const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for fitting a model for the calibration of
 * marginal probabilities.
 */
class IMarginalProbabilityCalibratorConfig {
    public:

        virtual ~IMarginalProbabilityCalibratorConfig() {};

        /**
         * Returns whether a holdout set should be used, if available, or not.
         *
         * @return True, if a holdout set should be used, false otherwise
         */
        virtual bool shouldUseHoldoutSet() const = 0;

        /**
         * Creates and returns a new object of template type `IMarginalProbabilityCalibratorFactory` according to the
         * configuration.
         *
         * @return An unique pointer to an object of template type `IMarginalProbabilityCalibratorFactory` that has been
         *         created
         */
        virtual std::unique_ptr<IMarginalProbabilityCalibratorFactory> createMarginalProbabilityCalibratorFactory()
          const = 0;
};
