/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/head_refinement/prediction.hpp"


/**
 * Defines an interface for all classes that allow to post-process the predictions of rules once they have been learned.
 */
class IPostProcessor {

    public:

        virtual ~IPostProcessor() { };

        /**
         * Post-processes the prediction of a rule.
         *
         * @param prediction A reference to an object of type `AbstractPrediction` that stores the predictions of a rule
         */
        virtual void postProcess(AbstractPrediction& prediction) const = 0;

};
