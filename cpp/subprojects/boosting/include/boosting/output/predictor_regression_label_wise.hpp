/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_regression.hpp"
#include "common/multi_threading/multi_threading.hpp"


namespace boosting {

    /**
     * Allows to configure predictors that predict label-wise regression scores for given query examples by summing up
     * the scores that are provided by the individual rules of an existing rule-based model for each label individually.
     */
    class LabelWiseRegressionPredictorConfig final : public IRegressionPredictorConfig {

        private:

            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

        public:

            /**
             * @param multiThreadingConfigPtr A reference to an unique pointer that stores the configuration of the
             *                                multi-threading behavior that should be used to predict for several query
             *                                examples in parallel
             */
            LabelWiseRegressionPredictorConfig(const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            /**
             * @see `IRegressionPredictorConfig::createRegressionPredictorFactory`
             */
            std::unique_ptr<IRegressionPredictorFactory> createRegressionPredictorFactory(
                const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;

    };

}
