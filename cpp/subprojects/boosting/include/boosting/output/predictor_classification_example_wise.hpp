/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_sparse.hpp"
#include "common/measures/measure_similarity.hpp"


namespace boosting {

    /**
     * Allows to predict the labels of given query examples using an existing rule-based model that has been learned
     * using a boosting algorithm.
     *
     * For prediction, the scores that are provided by the individual rules, are summed up. For each query example, the
     * aggregated score vector is then compared to known label vectors in order to obtain a distance measure. The label
     * vector that is closest to the aggregated score vector is finally predicted.
     */
    class ExampleWiseClassificationPredictor : public ISparsePredictor<uint8> {

        private:

            std::unique_ptr<ISimilarityMeasure> measurePtr_;

            uint32 numThreads_;

        public:

            /**
             * @param measurePtr    An unique pointer to an object of type `ISimilarityMeasure` that should be used to
             *                      quantify the similarity between predictions and known label vectors
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            ExampleWiseClassificationPredictor(std::unique_ptr<ISimilarityMeasure> measurePtr, uint32 numThreads);

            /**
             * Obtains predictions for different examples, based on predicted scores, and writes them to a given
             * prediction matrix.
             *
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the 
             *                          predicted scores
             * @param predictionMatrix  A reference to an object of type `CContiguousView`, the predictions should be
             *                          written to. May contain arbitrary values
             * @param labelVectors      A pointer to an object of type `LabelVectorSet` that stores all known label
             *                          vectors or a null pointer, if no such set is available
             */
            void transform(const CContiguousConstView<float64>& scoreMatrix,
                           CContiguousView<uint8>& predictionMatrix, const LabelVectorSet* labelVectors) const;

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
             * @see `ISparsePredictor::predictSparse`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
                const CContiguousFeatureMatrix& featureMatrix, uint32 numLabels, const RuleModel& model,
                const LabelVectorSet* labelVectors) const override;

            /**
             * @see `ISparsePredictor::predictSparse`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
                const CsrFeatureMatrix& featureMatrix, uint32 numLabels, const RuleModel& model,
                const LabelVectorSet* labelVectors) const override;

    };

}
