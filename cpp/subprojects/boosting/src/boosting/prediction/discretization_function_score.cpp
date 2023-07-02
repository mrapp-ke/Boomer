#include "boosting/prediction/discretization_function_score.hpp"

namespace boosting {

    /**
     * An implementation of the type `IDiscretizationFunction` that allows to discretize regression scores by comparing
     * them to a threshold.
     */
    class ScoreDiscretizationFunction final : public IDiscretizationFunction {
        private:

            float64 threshold_;

        public:

            /**
             * @param threshold The threshold that should be used for discretization
             */
            ScoreDiscretizationFunction(float64 threshold) : threshold_(threshold) {}

            bool discretizeScore(uint32 labelIndex, float64 score) const override {
                return score > threshold_;
            }
    };

    ScoreDiscretizationFunctionFactory::ScoreDiscretizationFunctionFactory(float64 threshold) : threshold_(threshold) {}

    std::unique_ptr<IDiscretizationFunction> ScoreDiscretizationFunctionFactory::create(
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const {
        return std::make_unique<ScoreDiscretizationFunction>(threshold_);
    }

}
