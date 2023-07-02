#include "common/rule_pruning/rule_pruning_irep.hpp"

/**
 * An implementation of the class `IRulePruning` that prunes rules by following the ideas of "incremental reduced error
 * pruning" (IREP).
 */
class Irep final : public IRulePruning {
    private:

        const RuleCompareFunction ruleCompareFunction_;

    public:

        /**
         * @param ruleCompareFunction An object of type `RuleCompareFunction` that defines the function that should be
         *                            used for comparing the quality of different rules
         */
        Irep(RuleCompareFunction ruleCompareFunction) : ruleCompareFunction_(ruleCompareFunction) {}

        std::unique_ptr<ICoverageState> prune(IThresholdsSubset& thresholdsSubset, IPartition& partition,
                                              ConditionList& conditions,
                                              const AbstractPrediction& head) const override {
            uint32 numConditions = conditions.getNumConditions();
            std::unique_ptr<ICoverageState> bestCoverageStatePtr;

            // Only rules with more than one condition can be pruned...
            if (numConditions > 1) {
                // Calculate the quality of the original rule on the prune set...
                const ICoverageState& originalCoverageState = thresholdsSubset.getCoverageState();
                Quality bestQuality = partition.evaluateOutOfSample(thresholdsSubset, originalCoverageState, head);

                // Create a copy of the original coverage mask...
                bestCoverageStatePtr = originalCoverageState.copy();

                // Reset the given thresholds...
                thresholdsSubset.resetThresholds();

                // We process the existing rule's conditions (except for the last one) in the order they have been
                // learned. At each iteration, we calculate the quality of a rule that only contains the conditions
                // processed so far and keep track of the best rule...
                ConditionList::const_iterator conditionIterator = conditions.cbegin();
                uint32 numPrunedConditions = 0;

                for (uint32 n = 1; n < numConditions; n++) {
                    // Filter the thresholds by applying the current condition...
                    const Condition& condition = *conditionIterator;
                    thresholdsSubset.filterThresholds(condition);

                    // Calculate the quality of a rule that contains the conditions that have been processed so far...
                    const ICoverageState& coverageState = thresholdsSubset.getCoverageState();
                    Quality quality = partition.evaluateOutOfSample(thresholdsSubset, coverageState, head);

                    // Check if the quality is better than the best quality seen so far (reaching the same quality with
                    // fewer conditions is considered an improvement)...
                    if (ruleCompareFunction_.compare(quality, bestQuality)
                        || (numPrunedConditions == 0 && !ruleCompareFunction_.compare(bestQuality, quality))) {
                        bestQuality = quality;
                        bestCoverageStatePtr = coverageState.copy();
                        numPrunedConditions = (numConditions - n);
                    }

                    conditionIterator++;
                }

                // Remove the pruned conditions...
                while (numPrunedConditions > 0) {
                    conditions.removeLastCondition();
                    numPrunedConditions--;
                }
            }

            return bestCoverageStatePtr;
        }
};

/**
 * Allows to create instances of the type `IRulePruning` that prune rules by following the ideas of "incremental reduced
 * error pruning" (IREP). Given `n` conditions in the order of their induction, IREP may remove up to `n - 1` trailing
 * conditions, depending on which of the resulting rules comes with the greatest improvement in terms of quality as
 * measured on the prune set.
 */
class IrepFactory final : public IRulePruningFactory {
    private:

        const RuleCompareFunction ruleCompareFunction_;

    public:

        /**
         * @param ruleCompareFunction An object of type `RuleCompareFunction` that defines the function that should be
         *                            used for comparing the quality of different rules
         */
        IrepFactory(RuleCompareFunction ruleCompareFunction) : ruleCompareFunction_(ruleCompareFunction) {}

        std::unique_ptr<IRulePruning> create() const override {
            return std::make_unique<Irep>(ruleCompareFunction_);
        }
};

IrepConfig::IrepConfig(RuleCompareFunction ruleCompareFunction) : ruleCompareFunction_(ruleCompareFunction) {}

std::unique_ptr<IRulePruningFactory> IrepConfig::createRulePruningFactory() const {
    return std::make_unique<IrepFactory>(ruleCompareFunction_);
}
