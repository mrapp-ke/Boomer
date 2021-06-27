/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/output/predictor.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that implement a transformation function that is applied to the scores that
     * are predicted for individual labels.
     */
    class ILabelWiseTransformationFunction {

        public:

            virtual ~ILabelWiseTransformationFunction() { };

            /**
             * Transforms the score that is predicted for an individual label.
             *
             * @param predictedScore    The predicted score
             * @return                  The result of the transformation
             */
            virtual float64 transform(float64 predictedScore) const = 0;

    };

    /**
     * Allows to transform the score that is predicted for an individual label into a probability by applying the
     * logistic sigmoid function.
     */
    class LogisticFunction : public ILabelWiseTransformationFunction {

        public:

            float64 transform(float64 predictedScore) const override;

    };

    /**
     * Allows to predict probabilities for given query examples, which estimate the chance of individual labels to be
     * relevant, using an existing rule-based model that has been learned using a boosting algorithm.
     *
     * For prediction, the scores that are provided by the individual rules, are summed up. The aggregated scores are
     * then transformed into probabilities in [0, 1] according to a certain transformation function that is applied to
     * the labels individually.
     */
    class LabelWiseProbabilityPredictor : public IPredictor<float64> {

        private:

            std::shared_ptr<ILabelWiseTransformationFunction> transformationFunctionPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param transformationFunctionPtr A shared pointer to an object of type `ILabelWiseTransformationFunction`
             *                                  that should be used to transform predicted scores into probabilities
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             */
            LabelWiseProbabilityPredictor(std::shared_ptr<ILabelWiseTransformationFunction> transformationFunctionPtr,
                                          uint32 numThreads);

            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const RuleModel& model) const override;

            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const RuleModel& model) const override;

    };

}
