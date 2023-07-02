/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/discretization_function.hpp"

namespace boosting {

    /**
     * Allow to create instances of the type `IDiscretizationFunction` that discretize regression scores by comparing
     * them to a threshold.
     */
    class ScoreDiscretizationFunctionFactory : public IDiscretizationFunctionFactory {
        private:

            float64 threshold_;

        public:

            /**
             * @param threshold The threshold that should be used for discretization
             */
            ScoreDiscretizationFunctionFactory(float64 threshold);

            std::unique_ptr<IDiscretizationFunction> create(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const override;
    };

}
