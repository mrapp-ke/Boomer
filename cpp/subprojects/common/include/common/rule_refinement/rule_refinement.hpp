/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_refinement/refinement.hpp"
#include <memory>


/**
 * Defines an interface for all classes that allow to find the best refinement of existing rules.
 */
class IRuleRefinement {

    public:

        virtual ~IRuleRefinement() { };

        /**
         * Finds the best refinement of an existing rule.
         *
         * @param currentHead A pointer to an object of type `AbstractEvaluatedPrediction`, representing the head of the
         *                    existing rule or a null pointer, if no rule exists yet
         */
        virtual void findRefinement(const AbstractEvaluatedPrediction* currentHead) = 0;

        /**
         * Returns the best refinement that has been found by the function `findRefinement`.
         *
         * @return An unique pointer to an object of type `Refinement` that stores information about the best refinement
         *         that has been found
         */
        virtual std::unique_ptr<Refinement> pollRefinement() = 0;

};
