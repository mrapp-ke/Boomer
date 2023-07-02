/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/rule_evaluation/score_vector_binned_dense.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"
#include "common/rule_refinement/prediction_evaluated.hpp"

/**
 * Allows to process the scores that are stored by an `IScoreVector` in order to convert them into the head of a rule,
 * represented by an `AbstractEvaluatedPrediction`.
 */
class ScoreProcessor {
    private:

        std::unique_ptr<AbstractEvaluatedPrediction>& headPtr_;

    public:

        /**
         * @param headPtr   A reference to an unique pointer of type `AbstractEvaluatedPrediction` that should be used
         *                  to store the rule head that is created by the processor
         */
        ScoreProcessor(std::unique_ptr<AbstractEvaluatedPrediction>& headPtr);

        /**
         * Processes the scores that are stored by a `DenseScoreVector<CompleteIndexVector>` in order to convert them
         * into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `DenseScoreVector<CompleteIndexVector>` that stores the
         *                    scores to be processed
         */
        void processScores(const DenseScoreVector<CompleteIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseScoreVector<PartialIndexVector>` in order to convert them
         * into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `DenseScoreVector<PartialIndexVector>` that stores the
         *                    scores to be processed
         */
        void processScores(const DenseScoreVector<PartialIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseBinnedScoreVector<CompleteIndexVector>` in order to convert
         * them into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `DenseBinnedScoreVector<CompleteIndexVector>` that stores
         *                    the scores to be processed
         */
        void processScores(const DenseBinnedScoreVector<CompleteIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `DenseBinnedScoreVector<PartialIndexVector>` in order to convert
         * them into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `DenseBinnedScoreVector<PartialIndexVector>` that stores
         *                    the scores to be processed
         */
        void processScores(const DenseBinnedScoreVector<PartialIndexVector>& scoreVector);

        /**
         * Processes the scores that are stored by a `IScoreVector` in order to convert them into the head of a rule.
         *
         * @param scoreVector A reference to an object of type `IScoreVector` that stores the scores to be processed
         */
        void processScores(const IScoreVector& scoreVector);
};
