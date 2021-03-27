/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/output/predictor.hpp"


namespace boosting {

    /**
     * Allows to predict regression scores for given query examples using an existing rule-based model that has been
     * learned using a boosting algorithm.
     *
     * For prediction, the scores that are provided by the individual rules, are summed up for each label individually.
     */
    class LabelWiseRegressionPredictor : public IPredictor<float64> {

        private:

            uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            LabelWiseRegressionPredictor(uint32 numThreads);

            /**
             * Obtains predictions for all examples in a C-contiguous matrix, using a single rule, and writes them to a
             * given prediction matrix.
             *
             * @param featureMatrix     A reference to an object of type `CContiguousFeatureMatrix` that stores the
             *                          feature values of the examples
             * @param predictionMatrix  A reference to an object of type `CContiguousView`, the predictions should be
             *                          written to. May contain arbitrary values
             * @param rule              A reference to an object of type `Rule` that should be used to obtain the
             *                          predictions
             */
            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const Rule& rule) const;

            /**
             * Obtains predictions for all examples in a sparse CSR matrix, using a single rule, and writes them to a
             * given prediction matrix.
             *
             * @param featureMatrix     A reference to an object of type `CsrFeatureMatrix` that stores the feature
             *                          values of the examples
             * @param predictionMatrix  A reference to an object of type `CContiguousView`, the predictions should be
             *                          written to. May contain arbitrary values
             * @param rule              A reference to an object of type `Rule` that should be used to obtain the
             *                          predictions
             */
            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const Rule& rule) const;

            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const RuleModel& model) const override;

            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const RuleModel& model) const override;

    };

}
