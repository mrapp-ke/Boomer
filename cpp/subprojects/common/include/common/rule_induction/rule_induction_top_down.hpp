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

        uint32 numThreads_;

    public:

        /**
         * @param numThreads The number of CPU threads to be used to search for potential refinements of a rule in
         *                   parallel. Must be at least 1
         */
        TopDownRuleInduction(uint32 numThreads);

        void induceDefaultRule(IStatisticsProvider& statisticsProvider,
                               const IHeadRefinementFactory* headRefinementFactory,
                               IModelBuilder& modelBuilder) const override;

        bool induceRule(IThresholds& thresholds, const IIndexVector& labelIndices, const IWeightVector& weights,
                        IPartition& partition, const IFeatureSubSampling& featureSubSampling, const IPruning& pruning,
                        const IPostProcessor& postProcessor, uint32 minCoverage, intp maxConditions,
                        intp maxHeadRefinements, RNG& rng, IModelBuilder& modelBuilder) const override;

};
