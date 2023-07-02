/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_evaluation/rule_compare_function.hpp"
#include "common/rule_refinement/refinement.hpp"
#include "common/rule_refinement/score_processor.hpp"

/**
 * Allows comparing potential refinements of a rule and keeping track of the best one.
 */
class SingleRefinementComparator final {
    private:

        const RuleCompareFunction ruleCompareFunction_;

        Refinement bestRefinement_;

        Quality bestQuality_;

        ScoreProcessor scoreProcessor_;

    public:

        /**
         * @param ruleCompareFunction An object of type `RuleCompareFunction` that defines the function that should be
         *                            used for comparing the quality of different rules
         */
        SingleRefinementComparator(RuleCompareFunction ruleCompareFunction);

        /**
         * @param comparator A reference to an object of type `SingleRefinementComparator` that keeps track of the best
         *                   refinement found so far
         */
        SingleRefinementComparator(const SingleRefinementComparator& comparator);

        /**
         * An iterator that provides access to the refinements the comparator keeps track of and allows to modify them.
         */
        typedef Refinement* iterator;

        /**
         * Returns an `iterator` to the beginning of the refinements, starting with the best one.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator to the worst end of the refinements.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Returns the number of refinements the comparator keeps track of.
         *
         * @return The number of refinements
         */
        uint32 getNumElements() const;

        /**
         * Returns whether the quality of a rule's predictions is considered as an improvement over the refinements that
         * have been provided to this comparator so far.
         *
         * @param scoreVector   A reference to an object of type `IScoreVector` that stores the quality of the
         *                      predictions
         * @return              True, if the quality of the given predictions is considered as an improvement, false
         *                      otherwise
         */
        bool isImprovement(const IScoreVector& scoreVector) const;

        /**
         * Keeps track of a given refinement of a rule that is considered as an improvement over the refinements that
         * have been provided to this comparator so far.
         *
         * @param refinement    A reference to an object of type `Refinement` that represents the refinement of the rule
         * @param scoreVector   A reference to an object of type `IScoreVector` that stores the predictions of the rule
         */
        void pushRefinement(const Refinement& refinement, const IScoreVector& scoreVector);

        /**
         * Keeps track of the best refinement that is stored by a given `SingleRefinementComparator` if it is considered
         * as an improvement over the best refinement that has been provided to this comparator.
         *
         * @param comparator    A reference to an object of type `SingleRefinementComparator` that should be merged
         * @return              True, if the best refinement that is stored by the given `comparator` is considered as
         *                      an improvement over the best refinement that has been provided to this comparator
         */
        bool merge(SingleRefinementComparator& comparator);
};
