/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss.hpp"
#include "boosting/macros.hpp"
#include "common/multi_threading/multi_threading.hpp"
#include "common/prediction/predictor_binary.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a predictor that predicts known label vectors for
     * given query examples by comparing the predicted regression scores or probability estimates to the label vectors
     * encountered in the training data.
     */
    class MLRLBOOSTING_API IExampleWiseBinaryPredictorConfig {
        public:

            virtual ~IExampleWiseBinaryPredictorConfig() {}

            /**
             * Returns whether binary predictions are derived from probability estimates rather than regression scores
             * or not.
             *
             * @return True, if binary predictions are derived from probability estimates rather than regression scores,
             *         false otherwise
             */
            virtual bool isBasedOnProbabilities() const = 0;

            /**
             * Sets whether binary predictions should be derived from probability estimates rather than regression
             * scores or not.
             *
             * @param basedOnProbabilities  True, if binary predictions should be derived from probability estimates
             *                              rather than regression scores, false otherwise
             * @return                      A reference to an object of type `IExampleWiseBinaryPredictorConfig` that
             *                              allows further configuration of the predictor
             */
            virtual IExampleWiseBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities) = 0;

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
             *                                        `IExampleWiseBinaryPredictorConfig` that allows further
             *                                        configuration of the predictor
             */
            virtual IExampleWiseBinaryPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) = 0;
    };

    /**
     * Allows to configure a predictor that predicts known label vectors for given query examples by comparing the
     * predicted regression scores or probability estimates to the label vectors encountered in the training data.
     */
    class ExampleWiseBinaryPredictorConfig final : public IExampleWiseBinaryPredictorConfig,
                                                   public IBinaryPredictorConfig {
        private:

            bool basedOnProbabilities_;

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
            ExampleWiseBinaryPredictorConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr,
                                             const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            bool isBasedOnProbabilities() const override;

            IExampleWiseBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities) override;

            bool isProbabilityCalibrationModelUsed() const override;

            IExampleWiseBinaryPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) override;

            /**
             * @see `IPredictorConfig::createPredictorFactory`
             */
            std::unique_ptr<IBinaryPredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                            uint32 numLabels) const override;

            /**
             * @see `IBinaryPredictorConfig::createSparsePredictorFactory`
             */
            std::unique_ptr<ISparseBinaryPredictorFactory> createSparsePredictorFactory(
              const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;
    };

}
