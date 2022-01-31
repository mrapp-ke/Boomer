/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_classification.hpp"
#include "common/multi_threading/multi_threading.hpp"
#include "boosting/losses/loss.hpp"


namespace boosting {

    /**
     * Allows to configure a predictor that automatically decides for a method that is used to predict whether
     * individual labels of given query examples are relevant or not
     */
    class AutomaticClassificationPredictorConfig : public IClassificationPredictorConfig {

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
            AutomaticClassificationPredictorConfig(
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
