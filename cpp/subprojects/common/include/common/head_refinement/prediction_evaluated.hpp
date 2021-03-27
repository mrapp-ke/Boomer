/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/head_refinement/prediction.hpp"


/**
 * An abstract base class for all classes that store the scores that are predicted by a rule, as well as a quality score
 * that assesses the overall quality of the rule.
 */
class AbstractEvaluatedPrediction : public AbstractPrediction {

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        AbstractEvaluatedPrediction(uint32 numElements);

        /**
         * A score that assesses the overall quality of the rule.
         */
        float64 overallQualityScore;

};