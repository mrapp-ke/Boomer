/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_induction/rule_induction.hpp"


/**
 * Allows to induce classification rules using a top-down greedy search, where new conditions are added iteratively to
 * the (initially empty) body of a rule. At each iteration, the refinement that improves the rule the most is chosen.
 * The search stops if no refinement results in an improvement.
 */
class TopDownRuleInduction : public IRuleInduction {

    private:

        uint32 minCoverage_;

        intp maxConditions_;

        intp maxHeadRefinements_;

        bool recalculatePredictions_;

        uint32 numThreads_;

    public:

        /**
         * @param minCoverage               The minimum number of training examples that must be covered by a rule. Must
         *                                  be at least 1
         * @param maxConditions             The maximum number of conditions to be included in a rule's body. Must be at
         *                                  least 1 or 0, if the number of conditions should not be restricted
         * @param maxHeadRefinements        The maximum number of times, the head of a rule may be refinement after a
         *                                  new condition has been added to its body. Must be at least 1 or 0, if the
         *                                  number of refinements should not be restricted
         * @param recalculatePredictions    True, if the predictions of rules should be recalculated on all training
         *                                  examples, if some of the examples have zero weights, false otherwise
         * @param numThreads                The number of CPU threads to be used to search for potential refinements of
         *                                  a rule in parallel. Must be at least 1
         */
        TopDownRuleInduction(uint32 minCoverage, uint32 maxConditions, uint32 maxHeadRefinements,
                             bool recalculatePredictions, uint32 numThreads);

        void induceDefaultRule(IStatistics& statistics, IModelBuilder& modelBuilder) const override;

        bool induceRule(IThresholds& thresholds, const IIndexVector& labelIndices, const IWeightVector& weights,
                        IPartition& partition, IFeatureSampling& featureSampling, const IPruning& pruning,
                        const IPostProcessor& postProcessor, RNG& rng, IModelBuilder& modelBuilder) const override;

};
