/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss.hpp"
#include "boosting/macros.hpp"
#include "common/multi_threading/multi_threading.hpp"
#include "common/prediction/predictor_probability.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a predictor that predicts label-wise probabilities
     * for given query examples by marginalizing over the joint probabilities of known label vectors.
     */
    class MLRLBOOSTING_API IMarginalizedProbabilityPredictorConfig {
        public:

            virtual ~IMarginalizedProbabilityPredictorConfig() {};

            /**
             * Returns whether a model for the calibration of probabilities is used, if available, or not.
             *
             * @return True, if a model for the calibration of probabilities is used, if available, false otherwise
             */
            virtual bool isProbabilityCalibrationModelUsed() const = 0;

            /**
             * Sets whether a model for the calibration of probabilities should be used, if available, or not.
             *
             * @param useProbabilityCalibrationModel  True, if a model for the calibration of probabilities should be
             *                                        used, if available, false otherwise
             * @return                                A reference to an object of type
             *                                        `IMarginalizedProbabilityPredictorConfig` that allows further
             *                                        configuration of the predictor
             */
            virtual IMarginalizedProbabilityPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) = 0;
    };

    /**
     * Allows to configure a predictor that predicts label-wise probabilities for given query examples by marginalizing
     * over the joint probabilities of known label vectors.
     */
    class MarginalizedProbabilityPredictorConfig final : public IMarginalizedProbabilityPredictorConfig,
                                                         public IProbabilityPredictorConfig {
        private:

            std::unique_ptr<IMarginalProbabilityCalibrationModel> noMarginalProbabilityCalibrationModelPtr_;

            std::unique_ptr<IJointProbabilityCalibrationModel> noJointProbabilityCalibrationModelPtr_;

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

        public:

            /**
             * @param lossConfigPtr             A reference to an unique pointer that stores the configuration of the
             *                                  loss function
             * @param multiThreadingConfigPtr   A reference to an unique pointer that stores the configuration of the
             *                                  multi-threading behavior that should be used to predict for several
             *                                  query examples in parallel
             */
            MarginalizedProbabilityPredictorConfig(
              const std::unique_ptr<ILossConfig>& lossConfigPtr,
              const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            bool isProbabilityCalibrationModelUsed() const override;

            IMarginalizedProbabilityPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) override;

            /**
             * @see `IProbabilityPredictorConfig::createPredictorFactory`
             */
            std::unique_ptr<IProbabilityPredictorFactory> createPredictorFactory(
              const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;
    };

}
