/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/discretization_function.hpp"
#include "boosting/prediction/probability_function_marginal.hpp"

namespace boosting {

    /**
     * Allow to create instances of the type `IDiscretizationFunction` that discretize regression scores by transforming
     * them into marginal probabilities.
     */
    class ProbabilityDiscretizationFunctionFactory : public IDiscretizationFunctionFactory {
        private:

            std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr_;

        public:

            /**
             * @param marginalProbabilityFunctionFactoryPtr An unique pointer to an object of type
             *                                              `IMarginalProbabilityFunctionFactory` that allows to create
             *                                              the implementation to be used to transform regression scores
             *                                              into marginal probabilities
             */
            ProbabilityDiscretizationFunctionFactory(
              std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr);

            std::unique_ptr<IDiscretizationFunction> create(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const override;
    };

}
