/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_classification.hpp"
#include "common/measures/measure_similarity.hpp"
#include "common/multi_threading/multi_threading.hpp"
#include "boosting/losses/loss.hpp"


namespace boosting {

    /**
     * Allows to configure a predictor that predicts known label vectors for given query examples by summing up the
     * scores that are provided by an existing rule-based model and comparing the aggregated score vector to the known
     * label vectors according to a certain distance measure. The label vector that is closest to the aggregated score
     * vector is finally predicted.
     */
    class ExampleWiseClassificationPredictorConfig final : public IClassificationPredictorConfig {

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
            ExampleWiseClassificationPredictorConfig(
                const std::unique_ptr<ILossConfig>& lossConfigPtr,
                const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            /**
             * @see `IClassificationPredictorConfig::createClassificationPredictorFactory`
             */
            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory(
                const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `IClassificationPredictorConfig::createLabelSpaceInfo`
             */
            std::unique_ptr<ILabelSpaceInfo> createLabelSpaceInfo(
                const IRowWiseLabelMatrix& labelMatrix) const override;

    };

}
