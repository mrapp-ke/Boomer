/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/output/predictor_sparse.hpp"


namespace boosting {

    /**
     * Allows to predict the labels of given query examples using an existing rule-based model that has been learned
     * using a boosting algorithm.
     *
     * For prediction, the scores that are provided by the individual rules, are summed up. The aggregated scores are
     * then transformed into binary values according to a certain threshold that is applied to the labels individually
     * (1 if a score exceeds the threshold, i.e., the label is relevant, 0 otherwise).
     */
    class LabelWiseClassificationPredictor : public ISparsePredictor<uint8> {

        private:

            float64 threshold_;

            uint32 numThreads_;

        public:

            /**
             * @param threshold     The threshold to be used
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            LabelWiseClassificationPredictor(float64 threshold, uint32 numThreads);

            /**
             * @see `IPredictor::predict`
             */
            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model, const LabelVectorSet* labelVectors) const override;

            /**
             * @see `IPredictor::predict`
             */
            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model, const LabelVectorSet* labelVectors) const override;

            /**
             * @see `ISparsePredictor::predict`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predict(const CContiguousFeatureMatrix& featureMatrix,
                                                                  uint32 numLabels, const RuleModel& model,
                                                                  const LabelVectorSet* labelVectors) const override;

            /**
             * @see `ISparsePredictor::predict`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predict(const CsrFeatureMatrix& featureMatrix,
                                                                  uint32 numLabels, const RuleModel& model,
                                                                  const LabelVectorSet* labelVectors) const override;

    };

}
