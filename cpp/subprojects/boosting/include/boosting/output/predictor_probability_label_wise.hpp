/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_probability.hpp"
#include "common/multi_threading/multi_threading.hpp"
#include "boosting/losses/loss.hpp"


namespace boosting {

    /**
     * Allows to configure a predictor that predicts label-wise probabilities for given query examples, which estimate
     * the chance of individual labels to be relevant, by summing up the scores that are provided by individual rules of
     * an existing rule-based models and transforming the aggregated scores into probabilities in [0, 1] according to a
     * certain transformation function that is applied to each label individually.
     */
    class LabelWiseProbabilityPredictorConfig final : public IProbabilityPredictorConfig {

        private:

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

            /**
             * @see `IProbabilityPredictorConfig::createProbabilityPredictorFactory`
             */
            std::unique_ptr<IProbabilityPredictorFactory> createProbabilityPredictorFactory(
                const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;

    };

}
