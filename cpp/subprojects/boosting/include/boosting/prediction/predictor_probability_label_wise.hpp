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
     * for given query examples by transforming the regression scores that are predicted for each label individually
     * into probabilities.
     */
    class MLRLBOOSTING_API ILabelWiseProbabilityPredictorConfig {
        public:

            virtual ~ILabelWiseProbabilityPredictorConfig() {};

            /**
             * Returns whether a model for the calibration of probabilities is used, if available, or not.
             *
             * @return True, if a model for the calibration of probabilities is used, if available, false otherwise
             */
            virtual bool isProbabilityCalibrationModelUsed() const = 0;

            /**
             * Sets whether a model for the calibration of probabilities should be used, if available, or not.
             *
             * @param useProbabilityCalibrationModel    True, if a model for the calibration of probabilities should be
             *                                          used, if available, false otherwise
             * @return                                  A reference to an object of type
             *                                          `ILabelWiseProbabilityPredictorConfig` that allows further
             *                                          configuration of the predictor
             */
            virtual ILabelWiseProbabilityPredictorConfig& setUseProbabilityCalibrationModel(
              bool useProbabilityCalibrationModel) = 0;
    };

    /**
     * Allows to configure a predictor that predicts label-wise probabilities for given query examples by transforming
     * the regression scores that are predicted for each label individually into probabilities.
     *
     * summing up the scores that are provided by individual rules of
     * an existing rule-based model and transforming the aggregated scores into probabilities in [0, 1] according to a
     * certain transformation function that is applied to each label individually.
     */
    class LabelWiseProbabilityPredictorConfig final : public ILabelWiseProbabilityPredictorConfig,
                                                      public IProbabilityPredictorConfig {
        private:

            std::unique_ptr<IMarginalProbabilityCalibrationModel> noMarginalProbabilityCalibrationModelPtr_;

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
            LabelWiseProbabilityPredictorConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr,
                                                const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            bool isProbabilityCalibrationModelUsed() const override;

            ILabelWiseProbabilityPredictorConfig& setUseProbabilityCalibrationModel(
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
