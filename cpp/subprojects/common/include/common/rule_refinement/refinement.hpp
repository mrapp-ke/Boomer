/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/model/condition.hpp"
#include "common/rule_refinement/prediction_evaluated.hpp"

/**
 * Stores the properties of a potential refinement of a rule.
 */
struct Refinement final : public Condition {
    public:

        /**
         * Assigns the properties of an existing refinement, except for the scores that are predicted by the refined
         * rule, to this refinement.
         *
         * @param refinement    A reference to the existing refinement
         * @return              A reference to the modified refinement
         */
        Refinement& operator=(const Refinement& refinement) {
            Condition::operator=(refinement);
            previous = refinement.previous;
            return *this;
        }

        /**
         * An unique pointer to an object of type `AbstractEvaluatedPrediction` that stores the scores that are
         * predicted by the refined rule, as well as its overall quality.
         */
        std::unique_ptr<AbstractEvaluatedPrediction> headPtr;

        /**
         * The index of the last element, e.g., example or bin, that has been processed when evaluating the refined
         * rule.
         */
        int64 previous;
};
