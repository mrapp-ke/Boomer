/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_refinement/refinement_comparator_fixed.hpp"
#include "common/rule_refinement/refinement_comparator_single.hpp"

/**
 * Defines an interface for all classes that allow to find the best refinement of existing rules.
 */
class IRuleRefinement {
    public:

        virtual ~IRuleRefinement() {};

        /**
         * Finds the best refinement of an existing rule.
         *
         * @param comparator    A reference to an object of type `SingleRefinementComparator` that is used to compare
         *                      the potential refinements
         * @param minCoverage   The minimum number of examples that must be covered by the refinement
         */
        virtual void findRefinement(SingleRefinementComparator& comparator, uint32 minCoverage) = 0;

        /**
         * Finds the best refinements of an existing rule.
         *
         * @param comparator    A reference to an object of type `MultiRefinementComparator` that is used to compare the
         *                      potential refinements
         * @param minCoverage   The minimum number of examples that must be covered by the refinements
         */
        virtual void findRefinement(FixedRefinementComparator& comparator, uint32 minCoverage) = 0;
};
