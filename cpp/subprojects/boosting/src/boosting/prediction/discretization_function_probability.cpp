#include "boosting/prediction/discretization_function_probability.hpp"

namespace boosting {

    /**
     * An implementation of the type `IDiscretizationFunction` that allows to discretize regression scores by
     * transforming them into marginal probabilities.
     */
    class ProbabilityDiscretizationFunction final : public IDiscretizationFunction {
        private:

            std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr_;

        public:

            /**
             * @param marginalProbabilityFunctionPtr An unique pointer to an object of type
             *                                       `IMarginalProbabilityFunction` that should be used to transform
             *                                       regression scores into marginal probabilities
             */
            ProbabilityDiscretizationFunction(
              std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr)
                : marginalProbabilityFunctionPtr_(std::move(marginalProbabilityFunctionPtr)) {}

            bool discretizeScore(uint32 labelIndex, float64 score) const override {
                float64 marginalProbability =
                  marginalProbabilityFunctionPtr_->transformScoreIntoMarginalProbability(labelIndex, score);
                return marginalProbability > 0.5;
            }
    };

    ProbabilityDiscretizationFunctionFactory::ProbabilityDiscretizationFunctionFactory(
      std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr)
        : marginalProbabilityFunctionFactoryPtr_(std::move(marginalProbabilityFunctionFactoryPtr)) {}

    std::unique_ptr<IDiscretizationFunction> ProbabilityDiscretizationFunctionFactory::create(
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const {
        std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr =
          marginalProbabilityFunctionFactoryPtr_->create(marginalProbabilityCalibrationModel);
        return std::make_unique<ProbabilityDiscretizationFunction>(std::move(marginalProbabilityFunctionPtr));
    }

}
