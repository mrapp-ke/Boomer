/**
 * Provides classes that allow to store gradients and Hessians that are calculated according to a (decomposable) loss
 * function that is applied label-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/statistics.h"
#include "label_wise_rule_evaluation.h"
#include "label_wise_losses.h"
#include "statistics.h"
#include <memory>


namespace boosting {

    /**
     * Allows to search for the best refinement of a rule based on the gradients and Hessians previously stored by an
     * object of type `DenseLabelWiseStatisticsImpl`.
     */
    class DenseLabelWiseRefinementSearchImpl : public AbstractDecomposableRefinementSearch {

        private:

            std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr_;

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

        public:

            /**
             * @param ruleEvaluationPtr     A shared pointer to an object of type `AbstractLabelWiseRuleEvaluation` to
             *                              be used for calculating the predictions, as well as corresponding quality
             *                              scores of rules
             * @param numPredictions        The number of labels to be considered by the search
             * @param labelIndices          A pointer to an array of type `uint32`, shape `(numPredictions)`,
             *                              representing the indices of the labels that should be considered by the
             *                              search or NULL, if all labels should be considered
             * @param numLabels             The total number of labels
             * @param gradients             A pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
             *                              representing the gradient for each example and label
             * @param totalSumsOfGradients  A pointer to an array of type `float64`, shape `(num_labels)`, representing
             *                              the sum of the gradients of all examples, which should be considered by the
             *                              search, for each label
             * @param hessians              A pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
             *                              representing the Hessian for each example and label
             * @param totalSumsOfHessians   A pointer to an array of type `float64`, shape `(num_labels)`, representing
             *                              the sum of the Hessians of all examples, which should be considered by the
             *                              search, for each label
             */
            DenseLabelWiseRefinementSearchImpl(std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr,
                                               uint32 numPredictions, const uint32* labelIndices, uint32 numLabels,
                                               const float64* gradients, const float64* totalSumsOfGradients,
                                               const float64* hessians, const float64* totalSumsOfHessians);

            ~DenseLabelWiseRefinementSearchImpl();

            void updateSearch(uint32 statisticIndex, uint32 weight) override;

            void resetSearch() override;

            LabelWisePredictionCandidate* calculateLabelWisePrediction(bool uncovered, bool accumulated) override;

    };

    /**
     * An abstract base class for all classes that store gradients and Hessians that are calculated according to a
     * differentiable loss function that is applied label-wise.
     */
    class AbstractLabelWiseStatistics : public AbstractGradientStatistics {

        public:

            std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr_;

            /**
             * @param numStatistics     The number of statistics
             * @param numLabels         The number of labels
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractLabelWiseRuleEvaluation`, to be
             *                          used for calculating the predictions, as well as corresponding quality scores,
             *                          of rules
             */
            AbstractLabelWiseStatistics(uint32 numStatistics, uint32 numLabels,
                                        std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr);

            /**
             * Sets the implementation to be used for calculating the predictions, as well as corresponding quality
             * scores, of rules.
             *
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractLabelWiseRuleEvaluation` to be
             *                          set
             */
            void setRuleEvaluation(std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr);

    };

    /**
     * Allows to store gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied label-wise using dense data structures.
     */
    class DenseLabelWiseStatisticsImpl : public AbstractLabelWiseStatistics {

        private:

            std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr_;

            std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr_;

            float64* currentScores_;

            float64* gradients_;

            float64* totalSumsOfGradients_;

            float64* hessians_;

            float64* totalSumsOfHessians_;

        public:

            /**
             * @param lossFunctionPtr   A shared pointer to an object of type `AbstractLabelWiseLoss`, representing the
             *                          loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractLabelWiseRuleEvaluation`, to be
             *                          used for calculating the predictions, as well as corresponding quality scores,
             *                          of rules
             * @param labelMatrixPtr    A shared pointer to an object of type `AbstractRandomAccessLabelMatrix` that
             *                          provides random access to the labels of the training examples
             * @param gradients         A pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
             *                          representing the gradients
             * @param hessians          A pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
             *                          representing the Hessians
             * @param current_scores    A pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
             *                          representing the currently predicted scores
             */
            DenseLabelWiseStatisticsImpl(std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr,
                                         std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr,
                                         std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr,
                                         float64* gradients, float64* hessians, float64* current_scores);

            ~DenseLabelWiseStatisticsImpl();

            void resetCoveredStatistics() override;

            void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override;

            AbstractRefinementSearch* beginSearch(uint32 numLabelIndices, const uint32* labelIndices) override;

            void applyPrediction(uint32 statisticIndex, Prediction* prediction) override;

    };

    /**
     * An abstract base class for all classes that allow to create new instances of the class
     * `AbstractLabelWiseStatistics`.
     */
    class AbstractLabelWiseStatisticsFactory {

        public:

            virtual ~AbstractLabelWiseStatisticsFactory();

            /**
             * Creates a new instance of the class `AbstractLabelWiseStatistics`.
             *
             * @return A pointer to an object of type `AbstractLabelWiseStatistics` that has been created
             */
            virtual AbstractLabelWiseStatistics* create();

    };

    /**
     * A factory that allows to create new instances of the class `DenseLabelWiseStatisticsImpl`.
     */
    class DenseLabelWiseStatisticsFactoryImpl : public AbstractLabelWiseStatisticsFactory {

        private:

            std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr_;

            std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr_;

            std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr_;

        public:

            /**
             * @param lossFunctionPtr   A shared pointer to an object of type `AbstractLabelWiseLoss`, representing the
             *                          loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractLabelWiseRuleEvaluation`, to be
             *                          used for calculating the predictions, as well as corresponding quality scores,
             *                          of rules
             * @param labelMatrixPtr    A shared pointer to an object of type `AbstractRandomAccessLabelMatrix` that
             *                          provides random access to the labels of the training examples
             */
            DenseLabelWiseStatisticsFactoryImpl(std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr,
                                                std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr,
                                                std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr);

            ~DenseLabelWiseStatisticsFactoryImpl();

            AbstractLabelWiseStatistics* create() override;

    };

}
