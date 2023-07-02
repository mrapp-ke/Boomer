#include "common/rule_induction/rule_induction_top_down_greedy.hpp"

#include "common/math/math.hpp"
#include "common/util/validation.hpp"
#include "rule_induction_common.hpp"
#include "rule_induction_top_down_common.hpp"

/**
 * An implementation of the type `IRuleInduction` that allows to induce individual rules by using a greedy top-down
 * search.
 */
class GreedyTopDownRuleInduction final : public AbstractRuleInduction {
    private:

        const RuleCompareFunction ruleCompareFunction_;

        const uint32 minCoverage_;

        const uint32 maxConditions_;

        const uint32 maxHeadRefinements_;

        const uint32 numThreads_;

    public:

        /**
         * @param ruleCompareFunction       An object of type `RuleCompareFunction` that defines the function that
         *                                  should be used for comparing the quality of different rules
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
        GreedyTopDownRuleInduction(RuleCompareFunction ruleCompareFunction, uint32 minCoverage, uint32 maxConditions,
                                   uint32 maxHeadRefinements, bool recalculatePredictions, uint32 numThreads)
            : AbstractRuleInduction(recalculatePredictions), ruleCompareFunction_(ruleCompareFunction),
              minCoverage_(minCoverage), maxConditions_(maxConditions), maxHeadRefinements_(maxHeadRefinements),
              numThreads_(numThreads) {}

    protected:

        std::unique_ptr<IThresholdsSubset> growRule(
          IThresholds& thresholds, const IIndexVector& labelIndices, const IWeightVector& weights,
          IPartition& partition, IFeatureSampling& featureSampling, RNG& rng,
          std::unique_ptr<ConditionList>& conditionListPtr,
          std::unique_ptr<AbstractEvaluatedPrediction>& headPtr) const override {
            // The label indices for which the next refinement of the rule may predict
            const IIndexVector* currentLabelIndices = &labelIndices;
            // A list that contains the conditions in the rule's body (in the order they have been learned)
            conditionListPtr = std::make_unique<ConditionList>();
            // The comparator that is used to keep track of the best refinement of the rule
            SingleRefinementComparator refinementComparator(ruleCompareFunction_);
            // Whether a refinement of the current rule has been found
            bool foundRefinement = true;

            // Create a new subset of the given thresholds...
            std::unique_ptr<IThresholdsSubset> thresholdsSubsetPtr = weights.createThresholdsSubset(thresholds);

            // Search for the best refinement until no improvement in terms of the rule's quality is possible anymore or
            // until the maximum number of conditions has been reached...
            while (foundRefinement && (maxConditions_ == 0 || conditionListPtr->getNumConditions() < maxConditions_)) {
                // Sample features...
                const IIndexVector& sampledFeatureIndices = featureSampling.sample(rng);

                // Search for the best refinement...
                foundRefinement = findRefinement(refinementComparator, *thresholdsSubsetPtr, sampledFeatureIndices,
                                                 *currentLabelIndices, minCoverage_, numThreads_);

                if (foundRefinement) {
                    Refinement& bestRefinement = *refinementComparator.begin();

                    // Sort the rule's predictions by the corresponding label indices...
                    bestRefinement.headPtr->sort();

                    // Filter the current subset of thresholds by applying the best refinement that has been found...
                    thresholdsSubsetPtr->filterThresholds(bestRefinement);

                    // Add the new condition...
                    conditionListPtr->addCondition(bestRefinement);

                    // Keep the labels for which the rule predicts, if the head should not be further refined...
                    if (maxHeadRefinements_ > 0 && conditionListPtr->getNumConditions() >= maxHeadRefinements_) {
                        currentLabelIndices = bestRefinement.headPtr.get();
                    }

                    // Abort refinement process if the rule is not allowed to cover less examples...
                    if (bestRefinement.numCovered <= minCoverage_) {
                        break;
                    }
                }
            }

            Refinement& bestRefinement = *refinementComparator.begin();
            headPtr = std::move(bestRefinement.headPtr);
            return thresholdsSubsetPtr;
        }
};

/**
 * Allows to create instances of the type `IRuleInduction` that induce individual rules by using a greedy top-down
 * search, where new conditions are added iteratively to the (initially empty) body of a rule. At each iteration, the
 * refinement that improves the rule the most is chosen. The search stops if no refinement results in an improvement.
 */
class GreedyTopDownRuleInductionFactory final : public IRuleInductionFactory {
    private:

        const RuleCompareFunction ruleCompareFunction_;

        const uint32 minCoverage_;

        const uint32 maxConditions_;

        const uint32 maxHeadRefinements_;

        const bool recalculatePredictions_;

        const uint32 numThreads_;

    public:

        /**
         * @param ruleCompareFunction       An object of type `RuleCompareFunction` that defines the function that
         *                                  should be used for comparing the quality of different rules
         * @param minCoverage               The minimum number of training examples that must be covered by a rule. Must
         *                                  be at least 1
         * @param maxConditions             The maximum number of conditions to be included in a rule's body. Must be at
         *                                  least 1 or 0, if the number of conditions should not be restricted
         * @param maxHeadRefinements        The maximum number of times, the head of a rule may be refined after a new
         *                                  condition has been added to its body. Must be at least 1 or 0, if the number
         *                                  of refinements should not be restricted
         * @param recalculatePredictions    True, if the predictions of rules should be recalculated on all training
         *                                  examples, if some of the examples have zero weights, false otherwise
         * @param numThreads                The number of CPU threads to be used to search for potential refinements of
         *                                  a rule in parallel. Must be at least 1
         */
        GreedyTopDownRuleInductionFactory(RuleCompareFunction ruleCompareFunction, uint32 minCoverage,
                                          uint32 maxConditions, uint32 maxHeadRefinements, bool recalculatePredictions,
                                          uint32 numThreads)
            : ruleCompareFunction_(ruleCompareFunction), minCoverage_(minCoverage), maxConditions_(maxConditions),
              maxHeadRefinements_(maxHeadRefinements), recalculatePredictions_(recalculatePredictions),
              numThreads_(numThreads) {}

        std::unique_ptr<IRuleInduction> create() const override {
            return std::make_unique<GreedyTopDownRuleInduction>(ruleCompareFunction_, minCoverage_, maxConditions_,
                                                                maxHeadRefinements_, recalculatePredictions_,
                                                                numThreads_);
        }
};

GreedyTopDownRuleInductionConfig::GreedyTopDownRuleInductionConfig(
  RuleCompareFunction ruleCompareFunction, const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
    : ruleCompareFunction_(ruleCompareFunction), minCoverage_(1), minSupport_(0.0f), maxConditions_(0),
      maxHeadRefinements_(1), recalculatePredictions_(true), multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

uint32 GreedyTopDownRuleInductionConfig::getMinCoverage() const {
    return minCoverage_;
}

IGreedyTopDownRuleInductionConfig& GreedyTopDownRuleInductionConfig::setMinCoverage(uint32 minCoverage) {
    assertGreaterOrEqual<uint32>("minCoverage", minCoverage, 1);
    minCoverage_ = minCoverage;
    return *this;
}

float32 GreedyTopDownRuleInductionConfig::getMinSupport() const {
    return minSupport_;
}

IGreedyTopDownRuleInductionConfig& GreedyTopDownRuleInductionConfig::setMinSupport(float32 minSupport) {
    if (minSupport != 0) assertGreater<float32>("minSupport", minSupport, 0);
    if (minSupport != 0) assertLess<float32>("minSupport", minSupport, 1);
    minSupport_ = minSupport;
    return *this;
}

uint32 GreedyTopDownRuleInductionConfig::getMaxConditions() const {
    return maxConditions_;
}

IGreedyTopDownRuleInductionConfig& GreedyTopDownRuleInductionConfig::setMaxConditions(uint32 maxConditions) {
    if (maxConditions != 0) assertGreaterOrEqual<uint32>("maxConditions", maxConditions, 1);
    maxConditions_ = maxConditions;
    return *this;
}

uint32 GreedyTopDownRuleInductionConfig::getMaxHeadRefinements() const {
    return maxHeadRefinements_;
}

IGreedyTopDownRuleInductionConfig& GreedyTopDownRuleInductionConfig::setMaxHeadRefinements(uint32 maxHeadRefinements) {
    if (maxHeadRefinements != 0) assertGreaterOrEqual<uint32>("maxHeadRefinements", maxHeadRefinements, 1);
    maxHeadRefinements_ = maxHeadRefinements;
    return *this;
}

bool GreedyTopDownRuleInductionConfig::arePredictionsRecalculated() const {
    return recalculatePredictions_;
}

IGreedyTopDownRuleInductionConfig& GreedyTopDownRuleInductionConfig::setRecalculatePredictions(
  bool recalculatePredictions) {
    recalculatePredictions_ = recalculatePredictions;
    return *this;
}

std::unique_ptr<IRuleInductionFactory> GreedyTopDownRuleInductionConfig::createRuleInductionFactory(
  const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    uint32 numExamples = featureMatrix.getNumRows();
    uint32 minCoverage;

    if (minSupport_ > 0) {
        minCoverage = calculateBoundedFraction(numExamples, minSupport_, minCoverage_, numExamples);
    } else {
        minCoverage = std::min(numExamples, minCoverage_);
    }

    uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
    return std::make_unique<GreedyTopDownRuleInductionFactory>(
      ruleCompareFunction_, minCoverage, maxConditions_, maxHeadRefinements_, recalculatePredictions_, numThreads);
}
