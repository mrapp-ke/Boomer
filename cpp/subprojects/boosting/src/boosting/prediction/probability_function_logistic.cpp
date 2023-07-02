#include "boosting/prediction/probability_function_logistic.hpp"

#include "boosting/math/math.hpp"

namespace boosting {

    /**
     * An implementation of the class `IMarginalProbabilityFunction` that transforms regression scores that are
     * predicted for individual labels into marginal probabilities via the logistic sigmoid function.
     */
    class LogisticFunction final : public IMarginalProbabilityFunction {
        private:

            const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel_;

        public:

            /**
             * @param marginalProbabilityCalibrationModel A reference to an object of type
             *                                            `IMarginalProbabilityCalibrationModel` that should be used for
             *                                            the calibration of marginal probabilities
             */
            LogisticFunction(const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel)
                : marginalProbabilityCalibrationModel_(marginalProbabilityCalibrationModel) {}

            float64 transformScoreIntoMarginalProbability(uint32 labelIndex, float64 score) const override {
                return marginalProbabilityCalibrationModel_.calibrateMarginalProbability(labelIndex,
                                                                                         logisticFunction(score));
            }
    };

    std::unique_ptr<IMarginalProbabilityFunction> LogisticFunctionFactory::create(
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const {
        return std::make_unique<LogisticFunction>(marginalProbabilityCalibrationModel);
    }

}
