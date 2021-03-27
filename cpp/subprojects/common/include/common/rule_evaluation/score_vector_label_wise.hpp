/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_evaluation/score_vector.hpp"

// Forward declarations
class ILabelWiseScoreProcessor;


/**
 * Defines an interface for all one-dimensional vectors that store the scores that may be predicted by a rule, as well
 * as corresponding quality scores that assess the quality of the predictions for individual labels and an overall
 * quality score that assesses the overall quality of the rule.
 */
class ILabelWiseScoreVector : virtual public IScoreVector {

    public:

        virtual ~ILabelWiseScoreVector() { };

        /**
         * Passes the scores to an `ILabelWiseScoreProcessor` in order to convert them into the head of a rule.
         *
         * @param bestHead       A reference to an object of type `AbstractEvaluatedPrediction`, representing the best
         *                       head that has been created so far
         * @param scoreProcessor A reference to an object of type `ILabelWiseScoreProcessor`, the scores should be
         *                       passed to
         * @return               A pointer to an object of type `AbstractEvaluatedPrediction` that has been created or a
         *                       null pointer if no object has been created
         */
        virtual const AbstractEvaluatedPrediction* processScores(const AbstractEvaluatedPrediction* bestHead,
                                                                 ILabelWiseScoreProcessor& scoreProcessor) const = 0;

};
