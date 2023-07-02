#include "boosting/prediction/probability_function_chain_rule.hpp"

#include "common/iterator/binary_forward_iterator.hpp"

namespace boosting {

    /**
     * An implementation of the class `IJointProbabilityFunction` that transforms regression scores that are
     * predicted for an example into joint probabilities by applying an `IMarginalProbabilityFunction` to each one and
     * calculating the product of the resulting marginal probabilities according to the probabilistic chain rule.
     */
    class ChainRule final : public IJointProbabilityFunction {
        private:

            const std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr_;

            const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel_;

        public:

            /**
             * @param marginalProbabilityFunctionPtr    An unique pointer to an object of type
             *                                          `IMarginalProbabilityFunction` to be used to transform
             *                                          regression scores into marginal probabilities
             * @param jointProbabilityCalibrationModel  A reference to an object of type
             *                                          `IJointProbabilityCalibrationModel` that should be used for the
             *                                          calibration of marginal probabilities
             */
            ChainRule(std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr,
                      const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel)
                : marginalProbabilityFunctionPtr_(std::move(marginalProbabilityFunctionPtr)),
                  jointProbabilityCalibrationModel_(jointProbabilityCalibrationModel) {}

            float64 transformScoresIntoJointProbability(
              uint32 labelVectorIndex, const LabelVector& labelVector,
              VectorConstView<float64>::const_iterator scoresBegin,
              VectorConstView<float64>::const_iterator scoresEnd) const override {
                auto labelIterator = make_binary_forward_iterator(labelVector.cbegin(), labelVector.cend());
                uint32 numLabels = scoresEnd - scoresBegin;
                float64 jointProbability = 1;

                for (uint32 i = 0; i < numLabels; i++) {
                    float64 score = scoresBegin[i];
                    float64 marginalProbability =
                      marginalProbabilityFunctionPtr_->transformScoreIntoMarginalProbability(i, score);
                    bool trueLabel = *labelIterator;

                    if (!trueLabel) {
                        marginalProbability = 1 - marginalProbability;
                    }

                    jointProbability *= marginalProbability;
                    labelIterator++;
                }

                return jointProbabilityCalibrationModel_.calibrateJointProbability(labelVectorIndex, jointProbability);
            }

            float64 transformScoresIntoJointProbability(uint32 labelVectorIndex, const LabelVector& labelVector,
                                                        SparseSetMatrix<float64>::const_row scores,
                                                        uint32 numLabels) const override {
                auto labelIterator = make_binary_forward_iterator(labelVector.cbegin(), labelVector.cend());
                float64 jointProbability = 1;

                for (uint32 i = 0; i < numLabels; i++) {
                    const IndexedValue<float64>* entry = scores[i];
                    float64 score = entry ? entry->value : 0;
                    float64 marginalProbability =
                      marginalProbabilityFunctionPtr_->transformScoreIntoMarginalProbability(i, score);
                    bool trueLabel = *labelIterator;

                    if (!trueLabel) {
                        marginalProbability = 1 - marginalProbability;
                    }

                    jointProbability *= marginalProbability;
                    labelIterator++;
                }

                return jointProbabilityCalibrationModel_.calibrateJointProbability(labelVectorIndex, jointProbability);
            }
    };

    ChainRuleFactory::ChainRuleFactory(
      std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr)
        : marginalProbabilityFunctionFactoryPtr_(std::move(marginalProbabilityFunctionFactoryPtr)) {}

    std::unique_ptr<IJointProbabilityFunction> ChainRuleFactory::create(
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
      const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const {
        return std::make_unique<ChainRule>(
          marginalProbabilityFunctionFactoryPtr_->create(marginalProbabilityCalibrationModel),
          jointProbabilityCalibrationModel);
    }

}
