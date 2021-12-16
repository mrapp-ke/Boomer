/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_evaluation/score_vector_dense.hpp"
#include "common/rule_evaluation/score_vector_binned_dense.hpp"
#include "common/rule_refinement/prediction_evaluated.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"


/**
 * Allows to process the scores that are stored by an `IScoreVector` in order to convert them into the head of a rule,
 * represented by an `AbstractEvaluatedPrediction`.
 */
class ScoreProcessor {

    private:

        std::unique_ptr<AbstractEvaluatedPrediction> headPtr_;

    public:

        /**
         * Processes the scores that are stored by a `DenseScoreVector<CompleteIndexVector>` in order to convert them
         * into the head of a rule.
         *
         * @param scoreVector   A reference to an object of type `DenseScoreVector<CompleteIndexVector>` that stores the
         *                      scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created
         */
        const AbstractEvaluatedPrediction* processScores(const DenseScoreVector<CompleteIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseScoreVector<PartialIndexVector>` in order to convert them
         * into the head of a rule.
         *
         * @param scoreVector   A reference to an object of type `DenseScoreVector<PartialIndexVector>` that stores the
         *                      scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created
         */
        const AbstractEvaluatedPrediction* processScores(const DenseScoreVector<PartialIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseBinnedScoreVector<CompleteIndexVector>` in order to convert
         * them into the head of a rule.
         *
         * @param scoreVector   A reference to an object of type `DenseBinnedScoreVector<CompleteIndexVector>` that
         *                      stores the scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created
         */
        const AbstractEvaluatedPrediction* processScores(
            const DenseBinnedScoreVector<CompleteIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseBinnedScoreVector<PartialIndexVector>` in order to convert
         * them into the head of a rule.
         *
         * @param scoreVector   A reference to an object of type `DenseBinnedScoreVector<PartialIndexVector>` that
         *                      stores the scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created
         */
        const AbstractEvaluatedPrediction* processScores(const DenseBinnedScoreVector<PartialIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `IScoreVector` in order to convert them into the head of a rule.
         *
         * @param scoreVector   A reference to an object of type `IScoreVector` that stores the scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created
         */
        const AbstractEvaluatedPrediction* processScores(const IScoreVector& scoreVector);

        /**
         * Returns the best head that has been found by the function `findHead.
         *
         * @return An unique pointer to an object of type `AbstractEvaluatedPrediction`, representing the best head that
         *         has been found
         */
        std::unique_ptr<AbstractEvaluatedPrediction> pollHead();

};
