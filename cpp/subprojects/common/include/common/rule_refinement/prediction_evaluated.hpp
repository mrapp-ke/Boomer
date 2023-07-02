/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_refinement/prediction.hpp"
#include "common/util/quality.hpp"

/**
 * An abstract base class for all classes that store the scores that are predicted by a rule, as well as a numerical
 * score that assesses the overall quality of the rule.
 */
class AbstractEvaluatedPrediction : public AbstractPrediction,
                                    public Quality {
    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        AbstractEvaluatedPrediction(uint32 numElements);
};
