/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/output/predictor.hpp"


namespace seco {

    /**
     * Allows to predict the labels of given query examples using an existing rule-based model that has been learned
     * using a separate-and-conquer algorithm.
     *
     * For prediction, the rules are processed in the order they have been learned. If a rule covers an example, its
     * prediction (1 if the label is relevant, 0 otherwise) is applied to the labels individually, if none of the
     * previous rules has already predicted for that particular example and label.
     */
    class LabelWiseClassificationPredictor : public IPredictor<uint8> {

        private:

            uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            LabelWiseClassificationPredictor(uint32 numThreads);

            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model) const override;

            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model) const override;

    };

}
