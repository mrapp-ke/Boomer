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
     * Defines an interface for all classes that allow to configure a predictor that predicts whether individual labels
     * of given query examples are relevant or irrelevant by discretizing the regression scores or probability estimates
     * that are predicted for each label according to the general F-measure maximizer (GFM).
     */
    class MLRLBOOSTING_API IGfmBinaryPredictorConfig {
        public:

            virtual ~IGfmBinaryPredictorConfig() {};

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
             * @return                                A reference to an object of type `IGfmBinaryPredictorConfig` that
             *                                        allows further configuration of the predictor
             */
            virtual IGfmBinaryPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) = 0;
    };

    /**
     * Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
     * irrelevant by discretizing the regression scores or probability estimates that are predicted for each label
     * according to the general F-measure maximizer (GFM).
     */
    class GfmBinaryPredictorConfig final : public IGfmBinaryPredictorConfig,
                                           public IBinaryPredictorConfig {
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
            GfmBinaryPredictorConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr,
                                     const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            bool isProbabilityCalibrationModelUsed() const override;

            IGfmBinaryPredictorConfig& setUseProbabilityCalibrationModel(bool useProbabilityCalibrationModel) override;

            /**
             * @see `IPredictorFactory::createPredictorFactory`
             */
            std::unique_ptr<IBinaryPredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                            uint32 numLabels) const override;

            /**
             * @see `IBinaryPredictorFactory::createSparsePredictorFactory`
             */
            std::unique_ptr<ISparseBinaryPredictorFactory> createSparsePredictorFactory(
              const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;
    };

}
