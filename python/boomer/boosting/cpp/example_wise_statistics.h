/**
 * Provides classes that allow to store gradients and Hessians that are calculated according to a (non-decomposable)
 * loss function that is applied example-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/statistics.h"
#include "example_wise_rule_evaluation.h"
#include "example_wise_losses.h"
#include "statistics.h"
#include "lapack.h"
#include <memory>


namespace boosting {

    /**
     * Allows to search for the best refinement of a rule based on the gradients and Hessians previously stored by an
    `* object of type `DenseExampleWiseStatisticsImpl`.
     */
    class DenseExampleWiseRefinementSearchImpl : public AbstractRefinementSearch {

        private:

            std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

            uint32 numPredictions_;

            const uint32* labelIndices_;

            uint32 numLabels_;

            const float64* gradients_;

            const float64* totalSumsOfGradients_;

            float64* sumsOfGradients_;

            float64* accumulatedSumsOfGradients_;

            const float64* hessians_;

            const float64* totalSumsOfHessians_;

            float64* sumsOfHessians_;

            float64* accumulatedSumsOfHessians_;

            LabelWisePredictionCandidate* prediction_;

            float64* tmpGradients_;

            float64* tmpHessians_;

            int dsysvLwork_;

            float64* dsysvTmpArray1_;

            int* dsysvTmpArray2_;

            double* dsysvTmpArray3_;

            float64* dspmvTmpArray_;

        public:

            /**
             * @param ruleEvaluationPtr     A shared pointer to an object of type `AbstractExampleWiseRuleEvaluation` to
             *                              be used for calculating the predictions, as well as corresponding quality
             *                              scores of rules
             * @param lapackPtr             A shared pointer to an object of type `Lapack` that allows to execute
             *                              different lapack routines
             * @param numPredictions        The number of labels to be considered by the search
             * @param labelIndices          A pointer to an array of type `uint32`, shape `(numPredictions)`,
             *                              representing the indices of the labels that should be considered by the
             *                              search or NULL, if all labels should be considered
             * @param numLabels             The total number of labels
             * @param gradients             A pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
             *                              representing the gradients for each example
             * @param totalSumsOfGradients  A pointer to an array of type `float64`, shape `(num_labels)`, representing
             *                              the sum of the gradients of all examples, which should be considered by the
             *                              search
             * @param hessians              A pointer to an array of type `float64`, shape
             *                              `(num_examples, (num_labels * (num_labels + 1)) / 2)`, representing the
             *                              Hessians for each example
             * @param totalSumsOfHessians   A pointer to an array of type `float64`, shape
             *                              `((num_labels * (num_labels + 1)) / 2)`, representing the sum of the
             *                              Hessians of all examples, which should be considered by the
             *                              search
             */
            DenseExampleWiseRefinementSearchImpl(std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr,
                                                 std::shared_ptr<Lapack> lapackPtr, uint32 numPredictions,
                                                 const uint32* labelIndices, uint32 numLabels, const float64* gradients,
                                                 const float64* totalSumsOfGradients, const float64* hessians,
                                                 const float64* totalSumsOfHessians);

            ~DenseExampleWiseRefinementSearchImpl();

            void updateSearch(uint32 statisticIndex, uint32 weight) override;

            void resetSearch() override;

            LabelWisePredictionCandidate* calculateLabelWisePrediction(bool uncovered, bool accumulated) override;

            PredictionCandidate* calculateExampleWisePrediction(bool uncovered, bool accumulated) override;

    };

    /**
     * An abstract base class for all classes that allow to store gradients and Hessians that are calculated according
     * to a differentiable loss function that is applied example-wise.
     */
    class AbstractExampleWiseStatistics : public AbstractGradientStatistics {

        public:

            std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr_;

            /**
             * @param numStatistics     The number of statistics
             * @param numLabels         The number of labels
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractExampleWiseRuleEvaluation`, to be
             *                          used for calculating the predictions, as well as corresponding quality scores,
             *                          of rules
             */
            AbstractExampleWiseStatistics(uint32 numStatistics, uint32 numLabels,
                                          std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr);

            /**
             * Sets the implementation to be used for calculating the predictions, as well as corresponding quality
             * scores, of rules.
             *
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractExampleWiseRuleEvaluation` to be
             *                          set
             */
            void setRuleEvaluation(std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr);

    };

    /**
     * Allows to store gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied example-wise using dense data structures.
     */
    class DenseExampleWiseStatisticsImpl : public AbstractExampleWiseStatistics {

        private:

            std::shared_ptr<AbstractExampleWiseLoss> lossFunctionPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

            std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr_;

            float64* currentScores_;

            float64* gradients_;

            float64* totalSumsOfGradients_;

            float64* hessians_;

            float64* totalSumsOfHessians_;

        public:

            /**
             * @param lossFunctionPtr   A shared pointer to an object of type `AbstractExampleWiseLoss`, representing
             *                          the loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractExampleWiseRuleEvaluation`, to be
             *                          used for calculating the predictions, as well as corresponding quality scores,
             *                          of rules
             * @param lapackPtr         A shared pointer to an object of type `Lapack` that allows to execute different
             *                          Lapack routines
             * @param labelMatrixPtr    A shared pointer to an object of type `AbstractRandomAccessLabelMatrix` that
             *                          provides random access to the labels of the training examples
             * @param gradients         A pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
             *                          representing the gradients
             * @param hessians          A pointer to an array of type `float64`, shape
             *                          `(num_examples, num_labels + (num_labels + 1) // 2)`, representing the Hessians
             * @param currentScores     A pointer to an array of type `float64`, shape `(num_examples, num_labels`),
             *                          representing the currently predicted scores
             */
            DenseExampleWiseStatisticsImpl(std::shared_ptr<AbstractExampleWiseLoss> lossFunctionPtr,
                                          std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr,
                                          std::shared_ptr<Lapack> lapackPtr,
                                          std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr,
                                          float64* gradients, float64* hessians, float64* currentScores);

            ~DenseExampleWiseStatisticsImpl();

            void resetCoveredStatistics() override;

            void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override;

            AbstractRefinementSearch* beginSearch(uint32 numLabelIndices, const uint32* labelIndices) override;

            void applyPrediction(uint32 statisticIndex, Prediction* prediction) override;

    };

    /**
     * An abstract base class for all classes that allow to create new instances of the class
     * `AbstractExampleWiseStatistics`.
     */
    class AbstractExampleWiseStatisticsFactory {

        public:

            virtual ~AbstractExampleWiseStatisticsFactory();

            /**
             * Creates a new instance of the class `AbstractExampleWiseStatistics`.
             *
             * @return A pointer to an object of type `AbstractExampleWiseStatistics` that has been created
             */
            virtual AbstractExampleWiseStatistics* create();

    };

    /**
     * A factory that allows to create new instances of the class `DenseExampleWiseStatisticsImpl`.
     */
    class DenseExampleWiseStatisticsFactoryImpl : public AbstractExampleWiseStatisticsFactory {

        private:

            std::shared_ptr<AbstractExampleWiseLoss> lossFunctionPtr_;

            std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

            std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr_;

        public:

            /**
             * @param lossFunctionPtr   A shared pointer to an object of type `AbstractExampleWiseLoss`, representing
             *                          the loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractExampleWiseRuleEvaluation`, to be
             *                          used for calculating the predictions, as well as corresponding quality scores,
             *                          of rules
             * @param lapackPtr         A shared pointer to an object of type `Lapack` that allows to execute different
             *                          Lapack routines
             * @param labelMatrixPtr    A shared pointer to an object of type `AbstractRandomAccessLabelMatrix` that
             *                          provides random access to the labels of the training examples
             */
            DenseExampleWiseStatisticsFactoryImpl(std::shared_ptr<AbstractExampleWiseLoss> lossFunctionPtr,
                                                  std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr,
                                                  std::shared_ptr<Lapack> lapackPtr,
                                                  std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr);

            ~DenseExampleWiseStatisticsFactoryImpl();

            AbstractExampleWiseStatistics* create() override;

    };

}
