/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/macros.hpp"
#include "boosting/rule_evaluation/regularization.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a regularization term that affects the evaluation of
     * rules by manually specifying the weight of the regularization term.
     */
    class MLRLBOOSTING_API IManualRegularizationConfig {
        public:

            virtual ~IManualRegularizationConfig() {};

            /**
             * Returns the weight of the regularization term.
             *
             * @return The weight of the regularization term
             */
            virtual float64 getRegularizationWeight() const = 0;

            /**
             * Sets the weight of the regularization term.
             *
             * @param regularizationWeight  The weight of the regularization term. Must be greater than 0
             * @return                      A reference to an object of type `IManualRegularizationConfig` that allows
             *                              further configuration of the regularization term
             */
            virtual IManualRegularizationConfig& setRegularizationWeight(float64 regularizationWeight) = 0;
    };

    /**
     * Allows to configure a regularization term that affects the evaluation of rules by manually specifying the weight
     * of the regularization term.
     */
    class ManualRegularizationConfig final : public IRegularizationConfig,
                                             public IManualRegularizationConfig {
        private:

            float64 regularizationWeight_;

        public:

            ManualRegularizationConfig();

            float64 getRegularizationWeight() const override;

            IManualRegularizationConfig& setRegularizationWeight(float64 regularizationWeight) override;

            float64 getWeight() const override;
    };

}
