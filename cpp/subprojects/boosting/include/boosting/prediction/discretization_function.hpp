/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/probability_calibration_marginal.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to discretize regression scores.
     */
    class IDiscretizationFunction {
        public:

            virtual ~IDiscretizationFunction() {};

            /**
             * Discretizes the regression score that is predicted for a specific label.
             *
             * @param labelIndex    The index of the label, the regression score is predicted for
             * @param score         The regression score to be discretized
             * @return              A binary value the given regression score has been turned into
             */
            virtual bool discretizeScore(uint32 labelIndex, float64 score) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `IDiscretizationFunction`.
     */
    class IDiscretizationFunctionFactory {
        public:

            virtual ~IDiscretizationFunctionFactory() {};

            /**
             * Creates and returns a new object of the type `IDiscretizationFunction`.
             *
             * @param marginalProbabilityCalibrationModel   A reference to an object of type
             *                                              `IMarginalProbabilityCalibrationModel` that should be used
             *                                              for the calibration of marginal probabilities
             * @return                                      An unique pointer to an object of type
             *                                              `IDiscretizationFunction` that has been created
             */
            virtual std::unique_ptr<IDiscretizationFunction> create(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const = 0;
    };

}
