/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_evaluation/score_vector_dense.hpp"
#include "common/indices/index_vector_full.hpp"
#include "common/indices/index_vector_partial.hpp"

// Forward declarations
class AbstractEvaluatedPrediction;


/**
 * Defines an interface for all classes that process the scores that are stored by an `IScoreVector` in order to convert
 * them into the head of a rule, represented by an `AbstractEvaluatedPrediction`.
 */
class IScoreProcessor {

    public:

        virtual ~IScoreProcessor() { };

        /**
         * Processes the scores that are stored by a `DenseScoreVector<FullIndexVector>` in order to convert them into
         * the head of a rule.
         *
         * @param bestHead      A pointer to an object of type `AbstractEvaluatedPrediction` that represents the best
         *                      head that has been created so far
         * @param scoreVector   A reference to an object of type `DenseScoreVector<FullIndexVector>` that stores the
         *                      scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created or a
         *                      null pointer if no object has been created
         */
        virtual const AbstractEvaluatedPrediction* processScores(
            const AbstractEvaluatedPrediction* bestHead, const DenseScoreVector<FullIndexVector>& scoreVector) = 0;

        /**
         * Processes the scores that are stored by a `DenseScoreVector<PartialIndexVector>` in order to convert them
         * into the head of a rule.
         *
         * @param bestHead      A pointer to an object of type `AbstractEvaluatedPrediction` that represents the best
         *                      head that has been created so far
         * @param scoreVector   A reference to an object of type `DenseScoreVector<PartialIndexVector>` that stores the
         *                      scores to be processed
         * @return              A pointer to an object of type `AbstractEvaluatedPrediction` that has been created or a
         *                      null pointer if no object has been created
         */
        virtual const AbstractEvaluatedPrediction* processScores(
            const AbstractEvaluatedPrediction* bestHead, const DenseScoreVector<PartialIndexVector>& scoreVector) = 0;

};
