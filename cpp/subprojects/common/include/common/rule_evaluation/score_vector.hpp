/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"

// Forward declarations
class IScoreProcessor;
class AbstractPrediction;
class AbstractEvaluatedPrediction;

/**
 * Defines an interface for all one-dimensional vectors that store the scores that may be predicted by a rule, as well
 * as a quality score that assess the overall quality of the rule.
 */
class IScoreVector {

    public:

        virtual ~IScoreVector() { };

        /**
         * A score that assesses the overall quality of the predicted score.
         */
        float64 overallQualityScore;

        /**
         * Sets the scores of a specific prediction to the scores that are stored in this vector.
         *
         * @param prediction A reference to an object of type `AbstractPrediction` that should be updated
         */
        virtual void updatePrediction(AbstractPrediction& prediction) const = 0;

        /**
         * Passes the scores to an `IScoreProcessor` in order to convert them into the head of a rule.
         *
         * @param bestHead       A reference to an object of type `AbstractEvaluatedPrediction`, representing the best
         *                       head that has been created so far
         * @param scoreProcessor A reference to an object of type `IScoreProcessor`, the scores should be passed to
         * @return               A pointer to an object of type `AbstractEvaluatedPrediction` that has been created or a
         *                       null pointer if no object has been created
         */
        virtual const AbstractEvaluatedPrediction* processScores(const AbstractEvaluatedPrediction* bestHead,
                                                                 IScoreProcessor& scoreProcessor) const = 0;

};
