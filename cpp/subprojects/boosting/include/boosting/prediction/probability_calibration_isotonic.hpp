/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss.hpp"
#include "boosting/macros.hpp"
#include "common/prediction/probability_calibration_isotonic.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a calibrator that fits a model for the calibration
     * of marginal probabilities via isotonic regression.
     */
    class MLRLBOOSTING_API IIsotonicMarginalProbabilityCalibratorConfig {
        public:

            virtual ~IIsotonicMarginalProbabilityCalibratorConfig() {};

            /**
             * Returns whether the calibration model is fit to the examples in the holdout set, if available, or not.
             *
             * @return True, if the calibration model is fit to the examples in the holdout set, if available, false
             *         if the training set is used instead
             */
            virtual bool isHoldoutSetUsed() const = 0;

            /**
             * Sets whether the calibration model should be fit to the examples in the holdout set, if available, or
             * not.
             *
             * @param useHoldoutSet True, if the calibration model should be fit to the examples in the holdout set, if
             *                      available, false if the training set should be used instead
             * @return              A reference to an object of type `IIsotonicMarginalProbabilityCalibratorConfig` that
             *                      allows further configuration of the calibrator
             */
            virtual IIsotonicMarginalProbabilityCalibratorConfig& setUseHoldoutSet(bool useHoldoutSet) = 0;
    };

    /**
     * Allows to configure a calibrator that fits a model for the calibration of marginal probabilities via isotonic
     * regression.
     */
    class IsotonicMarginalProbabilityCalibratorConfig final : public IIsotonicMarginalProbabilityCalibratorConfig,
                                                              public IMarginalProbabilityCalibratorConfig {
        private:

            bool useHoldoutSet_;

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

        public:

            /**
             * @param lossConfigPtr A reference to an unique pointer that stores the configuration of the loss function
             */
            IsotonicMarginalProbabilityCalibratorConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr);

            bool isHoldoutSetUsed() const override;

            IIsotonicMarginalProbabilityCalibratorConfig& setUseHoldoutSet(bool useHoldoutSet) override;

            /**
             * @see `IMarginalProbabilityCalibratorConfig::shouldUseHoldoutSet`
             */
            bool shouldUseHoldoutSet() const override;

            /**
             * @see `IMarginalProbabilityCalibratorConfig::createMarginalProbabilityCalibratorFactory`
             */
            std::unique_ptr<IMarginalProbabilityCalibratorFactory> createMarginalProbabilityCalibratorFactory()
              const override;
    };

    /**
     * Defines an interface for all classes that allow to configure a calibrator that fits a model for the calibration
     * of joint probabilities via isotonic regression.
     */
    class MLRLBOOSTING_API IIsotonicJointProbabilityCalibratorConfig {
        public:

            virtual ~IIsotonicJointProbabilityCalibratorConfig() {};

            /**
             * Returns whether the calibration model is fit to the examples in the holdout set, if available, or not.
             *
             * @return True, if the calibration model is fit to the examples in the holdout set, if available, false
             *         if the training set is used instead
             */
            virtual bool isHoldoutSetUsed() const = 0;

            /**
             * Sets whether the calibration model should be fit to the examples in the holdout set, if available, or
             * not.
             *
             * @param useHoldoutSet True, if the calibration model should be fit to the examples in the holdout set, if
             *                      available, false if the training set should be used instead
             * @return              A reference to an object of type `IIsotonicJointProbabilityCalibratorConfig` that
             *                      allows further configuration of the calibrator
             */
            virtual IIsotonicJointProbabilityCalibratorConfig& setUseHoldoutSet(bool useHoldoutSet) = 0;
    };

    /**
     * Allows to configure a calibrator that fits a model for the calibration of joint probabilities via isotonic
     * regression.
     */
    class IsotonicJointProbabilityCalibratorConfig final : public IIsotonicJointProbabilityCalibratorConfig,
                                                           public IJointProbabilityCalibratorConfig {
        private:

            bool useHoldoutSet_;

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

        public:

            /**
             * @param lossConfigPtr A reference to an unique pointer that stores the configuration of the loss function
             */
            IsotonicJointProbabilityCalibratorConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr);

            bool isHoldoutSetUsed() const override;

            IIsotonicJointProbabilityCalibratorConfig& setUseHoldoutSet(bool useHoldoutSet) override;

            /**
             * @see `IJointProbabilityCalibratorConfig::shouldUseHoldoutSet`
             */
            bool shouldUseHoldoutSet() const override;

            /**
             * @see `IJointProbabilityCalibratorConfig::isLabelVectorSetNeeeded`
             */
            bool isLabelVectorSetNeeded() const override;

            /**
             * @see `IJointProbabilityCalibratorConfig::createJointProbabilityCalibratorFactory`
             */
            std::unique_ptr<IJointProbabilityCalibratorFactory> createJointProbabilityCalibratorFactory()
              const override;
    };

}
