/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_evaluation/rule_compare_function.hpp"
#include "common/rule_evaluation/score_vector.hpp"
#include "common/rule_refinement/refinement.hpp"

#include <functional>
#include <vector>

/**
 * Allows comparing potential refinements of a rule and keeping track of the best ones.
 */
class FixedRefinementComparator final {
    private:

        const RuleCompareFunction ruleCompareFunction_;

        const uint32 maxRefinements_;

        Refinement* refinements_;

        std::vector<std::reference_wrapper<Refinement>> order_;

        Quality minQuality_;

    public:

        /**
         * @param ruleCompareFunction   An object of type `RuleCompareFunction` that defines the function that should be
         *                              used for comparing the quality of different rules
         * @param maxRefinements        The maximum number of refinements to keep track of
         * @param minQuality            A reference to an object of type `Quality` a refinement must improve on
         */
        FixedRefinementComparator(RuleCompareFunction ruleCompareFunction, uint32 maxRefinements,
                                  const Quality& minQuality);

        /**
         * @param ruleCompareFunction   An object of type `RuleCompareFunction` that defines the function that should be
         *                              used for comparing the quality of different rules
         * @param maxRefinements        The maximum number of refinements to keep track of
         */
        FixedRefinementComparator(RuleCompareFunction ruleCompareFunction, uint32 maxRefinements);

        /**
         * @param comparator A reference to an object of type `FixedRefinementComparator` that keeps track of the best
         *                   refinements found so far
         */
        FixedRefinementComparator(const FixedRefinementComparator& comparator);

        ~FixedRefinementComparator();

        /**
         * An iterator that provides access to the refinements the comparator keeps track of and allows to modify them.
         */
        typedef std::vector<std::reference_wrapper<Refinement>>::iterator iterator;

        /**
         * Returns an `iterator` to the beginning of the refinements, starting with the best one.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator to the end of the refinements.
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
         * Returns whether the quality of a rule's predictions is considered as an improvement over the quality of the
         * refinements that have been provided to this comparator so far.
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
         * Keeps track of the best refinements that are stored by a given `FixedRefinementComparator` if they are
         * considered as an improvement over the best refinements that have been provided to this comparator.
         *
         * @param comparator    A reference to an object of type `FixedRefinementComparator` that should be merged
         * @return              True, if at least one of the refinements that are stored by the given `comparator` is
         *                      considered as an improvement over the best refinements that has been provided to this
         *                      comparator
         */
        bool merge(FixedRefinementComparator& comparator);
};
