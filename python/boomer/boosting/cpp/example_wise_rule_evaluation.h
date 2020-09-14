/**
 * Implements classes for calculating the predictions of rules, as well as corresponding quality scores, based on the
 * gradients and Hessians that have been calculated according to a loss function that is applied example-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/predictions.h"
#include "blas.h"
#include "lapack.h"
#include <memory>


namespace boosting {

    /**
     * An abstract base class for all classes that allow to calculate the predictions of rule, as well as corresponding
     * quality scores, based on the gradients and Hessians that have been calculated according to a loss function that
     * is applied example-wise.
     */
    class AbstractExampleWiseRuleEvaluation {

        public:

            virtual ~AbstractExampleWiseRuleEvaluation();

            /**
             * Calculates the scores to be predicted by a rule, as well as corresponding quality scores, based on the
             * label-wise sums of gradients and Hessians that are covered by the rule. The predicted scores and quality
             * scores are stored in a given object of type `LabelWisePredictionCandidate`.
             *
             * If the argument `uncovered` is True, the rule is considered to cover the difference between the sums of
             * gradients and Hessians that are stored in the arrays `totalSumsOfGradients` and `sumsOfGradients` and
             * `totalSumsOfHessians` and `sumsOfHessians`, respectively.
             *
             * @param labelIndices          A pointer to an array of type `uint32`, shape
             *                              `(prediction.numPredictions_)`, representing the indices of the labels for
             *                              which the rule should predict or NULL, if the rule should predict for all
             *                              labels
             * @param totalSumsOfGradients  A pointer to an array of type `float64`, shape `(num_labels), representing
             *                              the total sums of gradients for individual labels
             * @param sumsOfGradients       A pointer to an array of type `float64`, shape
             *                              `(prediction.numPredictions_)`, representing the sums of gradients for
             *                              individual labels
             * @param totalSumsOfHessians   A pointer to an array of type `float64`, shape
             *                              `((num_labels * (num_labels + 1)) / 2)`, representing the total sums of
             *                              Hessians for individual labels
             * @param sumsOfHessians        A pointer to an array of type `float64`, shape
             *                              `(prediction.numPredictions_ * (prediction.numPredictions_ + 1) / 2)`,
             *                              representing the sums of Hessians for individual labels
             * @param uncovered             False, if the rule covers the sums of gradient and Hessians that are stored
             *                              in the array `sumsOfGradients` and `sumsOfHessians`, True, if the rule
             *                              covers the difference between the sums of gradients and Hessians that are
             *                              stored in the arrays `totalSumsOfGradients` and `sumsOfGradients` and
             *                              `totalSumsOfHessians` and `sumsOfHessians`, respectively
             * @param prediction            A pointer to an object of type `LabelWisePredictionCandidate` that should be
             *                              used to store the predicted scores and quality scores
             */
            virtual void calculateLabelWisePrediction(const uint32* labelIndices, const float64* totalSumsOfGradients,
                                                      float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                                      float64* sumsOfHessians, bool uncovered,
                                                      LabelWisePredictionCandidate* prediction);

            /**
             * Calculates the scores to be predicted by a rule, as well as an overall quality score, based on the sums
             * of gradients and Hessians that are covered by the rule. The predicted scores and quality scores are
             * stored in a given object of type `PredictionCandidate`.
             *
             * If the argument `uncovered` is True, the rule is considered to cover the difference between the sums of
             * gradients and Hessians that are stored in the arrays `totalSumsOfGradients` and `sumsOfGradients` and
             * `totalSumsOfHessians` and `sumsOfHessians`, respectively.
             *
             * @param labelIndices          A pointer to an array of type `uint32`, shape
             *                              `(prediction.numPredictions_)`, representing the indices of the labels for
             *                              which the rule should predict or NULL, if the rule should predict for all
             *                              labels
             * @param totalSumsOfGradients  A pointer to an array of type `float64`, shape `(num_labels), representing
             *                              the total sums of gradients for individual labels
             * @param sumsOfGradients       A pointer to an array of type `float64`, shape
             *                              `(prediction.numPredictions_)`, representing the sums of gradients for
             *                              individual labels
             * @param totalSumsOfHessians   A pointer to an array of type `float64`, shape
             *                              `((num_Labels * (num_labels + 1)) / 2)`, representing the total sums of
             *                              Hessians for individual labels
             * @param sumsOfHessians        A pointer to an array of type `float64`, shape
             *                              `(prediction.numPredictions_ * (prediction.numPredictions_ + 1) / 2)`,
             *                              representing the sums of Hessians for individual labels
             * @param tmpGradients          A pointer to an array of type `float64`, shape `(num_labels)` that will be
             *                              used to temporarily store gradients. May contain arbitrary values
             * @param tmpHessians           A pointer to an array of type `float64`, shape
             *                              `(prediction.numPredictions_ * (prediction.numPredictions_ + 1) / 2)` that
             *                              will be used to temporarily store Hessians. May contain arbitrary values
             * @param dsysvLwork            The value for the parameter "lwork" to be used by Lapack's DSYSV routine
             * @param dsysvTmpArray1        A pointer to an array of type `float64`, shape
             *                              `(prediction.numPredictions_, prediction.numPredictions_)` that will be used
             *                              to temporarily store values computed by Lapack's DSYSV routine. May contain
             *                              arbitrary values
             * @param dsysvTmpArray2        A pointer to an array of type `int`, shape `(prediction.numPredictions_)`
             *                              that will be used to temporarily store values computed by Lapack's DSYSV
             *                              routine. May contain arbitrary values
             * @param dsysvTmpArray3        A pointer to an array of type `double`, shape `(lwork)` that will be used to
             *                              temporarily store values computed by Lapack's DSYSV routine. May contain
             *                              arbitrary values
             * @param dspmvTmpArray         A pointer to an array of type `float64`, shape
             *                              `(prediction.numPredictions_)` that will be used to temporarily store values
             *                              computed by Blas' DSPMV routine. May contain arbitrary values
             * @param uncovered             False, if the rule covers the sums of gradient and Hessians that are stored
             *                              in the array `sumsOfGradients` and `sumsOfHessians`, True, if the rule
             *                              covers the difference between the sums of gradients and Hessians that are
             *                              stored in the arrays `totalSumsOfGradients` and `sumsOfGradients` and
             *                              `totalSumsOfHessians` and `sumsOfHessians`, respectively
             * @param prediction            A pointer to an object of type `PredictionCandidate` that should be used to
             *                              store the predicted scores and quality score
             */
            virtual void calculateExampleWisePrediction(const uint32* labelIndices, const float64* totalSumsOfGradients,
                                                        float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                                        float64* sumsOfHessians, float64* tmpGradients,
                                                        float64* tmpHessians, int dsysvLwork, float64* dsysvTmpArray1,
                                                        int* dsysvTmpArray2, double* dsysvTmpArray3,
                                                        float64* dspmvTmpArray, bool uncovered,
                                                        PredictionCandidate* prediction);

    };

    /**
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
     * Hessians that have been calculated according to a loss function that is applied example wise using L2
     * regularization.
     */
    class RegularizedExampleWiseRuleEvaluationImpl : public AbstractExampleWiseRuleEvaluation {

        private:

            float64 l2RegularizationWeight_;

            std::shared_ptr<Lapack> lapackPtr_;

            std::shared_ptr<Blas> blasPtr_;

        public:

            /**
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param blasPtr                   A shared pointer to an object of type `Blas` that allows to execute
             *                                  different BLAS routines
             * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
             *                                  different LAPACK routines
             */
            RegularizedExampleWiseRuleEvaluationImpl(float64 l2RegularizationWeight, std::shared_ptr<Blas> blasPtr,
                                                     std::shared_ptr<Lapack> lapackPtr);

            ~RegularizedExampleWiseRuleEvaluationImpl();

            void calculateLabelWisePrediction(const uint32* labelIndices, const float64* totalSumsOfGradients,
                                              float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                              float64* sumsOfHessians, bool uncovered,
                                              LabelWisePredictionCandidate* prediction) override;

            void calculateExampleWisePrediction(const uint32* labelIndices, const float64* totalSumsOfGradients,
                                                float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                                float64* sumsOfHessians, float64* tmpGradients, float64* tmpHessians,
                                                int dsysvLwork, float64* dsysvTmpArray1, int* dsysvTmpArray2,
                                                double* dsysvTmpArray3, float64* dspmvTmpArray, bool uncovered,
                                                PredictionCandidate* prediction) override;

    };

}
