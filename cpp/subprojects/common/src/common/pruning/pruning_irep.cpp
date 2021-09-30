#include "common/pruning/pruning_irep.hpp"


std::unique_ptr<ICoverageState> IREP::prune(IThresholdsSubset& thresholdsSubset, IPartition& partition,
                                            ConditionList& conditions, const AbstractPrediction& head) const {
    ConditionList::size_type numConditions = conditions.getNumConditions();
    std::unique_ptr<ICoverageState> bestCoverageStatePtr;

    // Only rules with more than one condition can be pruned...
    if (numConditions > 1) {
        // Calculate the quality score of the original rule on the prune set...
        const ICoverageState& originalCoverageState = thresholdsSubset.getCoverageState();
        float64 bestQualityScore = partition.evaluateOutOfSample(thresholdsSubset, originalCoverageState, head);

        // Create a copy of the original coverage mask...
        bestCoverageStatePtr = originalCoverageState.copy();

        // Reset the given thresholds...
        thresholdsSubset.resetThresholds();

        // We process the existing rule's conditions (except for the last one) in the order they have been learned. At
        // each iteration, we calculate the quality score of a rule that only contains the conditions processed so far
        // and keep track of the best rule...
        ConditionList::const_iterator conditionIterator = conditions.cbegin();
        ConditionList::size_type numPrunedConditions = 0;

        for (std::list<Condition>::size_type n = 1; n < numConditions; n++) {
            // Filter the thresholds by applying the current condition...
            const Condition& condition = *conditionIterator;
            thresholdsSubset.filterThresholds(condition);

            // Calculate the quality score of a rule that contains the conditions that have been processed so far...
            const ICoverageState& coverageState = thresholdsSubset.getCoverageState();
            float64 qualityScore = partition.evaluateOutOfSample(thresholdsSubset, coverageState, head);

            // Check if the quality score is better than the best quality score known so far (reaching the same score
            // with fewer conditions is considered an improvement)...
            if (qualityScore < bestQualityScore || (numPrunedConditions == 0 && qualityScore == bestQualityScore)) {
                bestQualityScore = qualityScore;
                bestCoverageStatePtr = coverageState.copy();
                numPrunedConditions = (numConditions - n);
            }

            conditionIterator++;
        }

        // Remove the pruned conditions...
        while (numPrunedConditions > 0) {
            conditions.removeLast();
            numPrunedConditions--;
        }
    }

    return bestCoverageStatePtr;
}
