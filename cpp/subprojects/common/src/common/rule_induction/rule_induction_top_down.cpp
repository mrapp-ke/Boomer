#include "common/rule_induction/rule_induction_top_down.hpp"
#include "common/rule_refinement/score_processor.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/util/validation.hpp"
#include "omp.h"
#include <unordered_map>


/**
 * An implementation of the type `IRuleInduction` that allows to induce classification rules by using a top-down greedy
 * search.
 */
class TopDownRuleInduction final : public IRuleInduction {

    private:

        uint32 minCoverage_;

        uint32 maxConditions_;

        uint32 maxHeadRefinements_;

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
                             bool recalculatePredictions, uint32 numThreads)
            : minCoverage_(minCoverage), maxConditions_(maxConditions), maxHeadRefinements_(maxHeadRefinements),
              recalculatePredictions_(recalculatePredictions), numThreads_(numThreads) {

        }

        void induceDefaultRule(IStatistics& statistics, IModelBuilder& modelBuilder) const override {
            uint32 numStatistics = statistics.getNumStatistics();
            uint32 numLabels = statistics.getNumLabels();
            statistics.resetSampledStatistics();

            for (uint32 i = 0; i < numStatistics; i++) {
                statistics.addSampledStatistic(i, 1);
            }

            CompleteIndexVector labelIndices(numLabels);
            std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = labelIndices.createSubset(statistics);
            const IScoreVector& scoreVector = statisticsSubsetPtr->calculatePrediction(true, false);
            ScoreProcessor scoreProcessor;
            scoreProcessor.processScores(scoreVector);
            std::unique_ptr<AbstractEvaluatedPrediction> defaultPredictionPtr = scoreProcessor.pollHead();

            for (uint32 i = 0; i < numStatistics; i++) {
                defaultPredictionPtr->apply(statistics, i);
            }

            modelBuilder.setDefaultRule(*defaultPredictionPtr);
        }

        bool induceRule(IThresholds& thresholds, const IIndexVector& labelIndices, const IWeightVector& weights,
                        IPartition& partition, IFeatureSampling& featureSampling, const IPruning& pruning,
                        const IPostProcessor& postProcessor, RNG& rng, IModelBuilder& modelBuilder) const override {
            // True, if the rule is learned on a sample of the available training examples, False otherwise
            bool instanceSamplingUsed = weights.hasZeroWeights();
            // The label indices for which the next refinement of the rule may predict
            const IIndexVector* currentLabelIndices = &labelIndices;
            // A (stack-allocated) list that contains the conditions in the rule's body (in the order they have been
            // learned)
            ConditionList conditions;
            // The total number of conditions
            uint32 numConditions = 0;
            // A map that stores a pointer to an object of type `IRuleRefinement` for each feature
            std::unordered_map<uint32, std::unique_ptr<IRuleRefinement>> ruleRefinements;
            std::unordered_map<uint32, std::unique_ptr<IRuleRefinement>>* ruleRefinementsPtr = &ruleRefinements;
            // An unique pointer to the best refinement of the current rule
            std::unique_ptr<Refinement> bestRefinementPtr = std::make_unique<Refinement>();
            // A pointer to the head of the best rule found so far
            AbstractEvaluatedPrediction* bestHead = nullptr;
            // Whether a refinement of the current rule has been found
            bool foundRefinement = true;

            // Create a new subset of the given thresholds...
            std::unique_ptr<IThresholdsSubset> thresholdsSubsetPtr = thresholds.createSubset(weights);

            // Search for the best refinement until no improvement in terms of the rule's quality score is possible
            // anymore or the maximum number of conditions has been reached...
            while (foundRefinement && (maxConditions_ == 0 || numConditions < maxConditions_)) {
                foundRefinement = false;

                // Sample features...
                const IIndexVector& sampledFeatureIndices = featureSampling.sample(rng);
                uint32 numSampledFeatures = sampledFeatureIndices.getNumElements();

                // For each feature, create an object of type `IRuleRefinement`...
                for (int64 i = 0; i < numSampledFeatures; i++) {
                    uint32 featureIndex = sampledFeatureIndices.getIndex((uint32) i);
                    std::unique_ptr<IRuleRefinement> ruleRefinementPtr = currentLabelIndices->createRuleRefinement(
                        *thresholdsSubsetPtr, featureIndex);
                    ruleRefinements[featureIndex] = std::move(ruleRefinementPtr);
                }

                // Search for the best condition among all available features to be added to the current rule...
                #pragma omp parallel for firstprivate(numSampledFeatures) firstprivate(ruleRefinementsPtr) \
                firstprivate(bestHead) schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numSampledFeatures; i++) {
                    uint32 featureIndex = sampledFeatureIndices.getIndex((uint32) i);
                    std::unique_ptr<IRuleRefinement>& ruleRefinementPtr =
                        ruleRefinementsPtr->find(featureIndex)->second;
                    ruleRefinementPtr->findRefinement(bestHead);
                }

                // Pick the best refinement among the refinements that have been found for the different features...
                for (int64 i = 0; i < numSampledFeatures; i++) {
                    uint32 featureIndex = sampledFeatureIndices.getIndex((uint32) i);
                    std::unique_ptr<IRuleRefinement>& ruleRefinementPtr = ruleRefinements.find(featureIndex)->second;
                    std::unique_ptr<Refinement> refinementPtr = ruleRefinementPtr->pollRefinement();

                    if (refinementPtr->isBetterThan(*bestRefinementPtr)) {
                        bestRefinementPtr = std::move(refinementPtr);
                        foundRefinement = true;
                    }
                }

                if (foundRefinement) {
                    bestHead = bestRefinementPtr->headPtr.get();

                    // Filter the current subset of thresholds by applying the best refinement that has been found...
                    thresholdsSubsetPtr->filterThresholds(*bestRefinementPtr);
                    uint32 numCoveredExamples = bestRefinementPtr->numCovered;

                    // Add the new condition...
                    conditions.addCondition(*bestRefinementPtr);
                    numConditions++;

                    // Keep the labels for which the rule predicts, if the head should not be further refined...
                    if (maxHeadRefinements_ > 0 && numConditions >= maxHeadRefinements_) {
                        currentLabelIndices = bestHead;
                    }

                    // Abort refinement process if the rule is not allowed to cover less examples...
                    if (numCoveredExamples <= minCoverage_) {
                        break;
                    }
                }
            }

            if (bestHead) {
                if (instanceSamplingUsed) {
                    // Prune rule...
                    IStatisticsProvider& statisticsProvider = thresholds.getStatisticsProvider();
                    statisticsProvider.switchToPruningRuleEvaluation();
                    std::unique_ptr<ICoverageState> coverageStatePtr = pruning.prune(*thresholdsSubsetPtr, partition,
                                                                                     conditions, *bestHead);
                    statisticsProvider.switchToRegularRuleEvaluation();

                    // Re-calculate the scores in the head based on the entire training data...
                    if (recalculatePredictions_) {
                        const ICoverageState& coverageState =
                            coverageStatePtr ? *coverageStatePtr : thresholdsSubsetPtr->getCoverageState();
                        partition.recalculatePrediction(*thresholdsSubsetPtr, coverageState, *bestRefinementPtr);
                    }
                }

                // Apply post-processor...
                postProcessor.postProcess(*bestHead);

                // Update the statistics by applying the predictions of the new rule...
                thresholdsSubsetPtr->applyPrediction(*bestHead);

                // Add the induced rule to the model...
                modelBuilder.addRule(conditions, *bestHead);
                return true;
            } else {
                // No rule could be induced, because no useful condition could be found. This might be the case, if all
                // examples have the same values for the considered features.
                return false;
            }
        }

};

/**
 * Allows to create instances of the type `IRuleInduction` that induce classification rules by using a top-down greedy
 * search, where new conditions are added iteratively to the (initially empty) body of a rule. At each iteration, the
 * refinement that improves the rule the most is chosen. The search stops if no refinement results in an improvement.
 */
class TopDownRuleInductionFactory final : public IRuleInductionFactory {

    private:

        uint32 minCoverage_;

        uint32 maxConditions_;

        uint32 maxHeadRefinements_;

        bool recalculatePredictions_;

        uint32 numThreads_;

    public:

        /**
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
        TopDownRuleInductionFactory(uint32 minCoverage, uint32 maxConditions, uint32 maxHeadRefinements,
                                    bool recalculatePredictions, uint32 numThreads)
            : minCoverage_(minCoverage), maxConditions_(maxConditions), maxHeadRefinements_(maxHeadRefinements),
              recalculatePredictions_(recalculatePredictions), numThreads_(numThreads) {

        }

        std::unique_ptr<IRuleInduction> create() const override {
            return std::make_unique<TopDownRuleInduction>(minCoverage_, maxConditions_, maxHeadRefinements_,
                                                          recalculatePredictions_, numThreads_);
        }

};


TopDownRuleInductionConfig::TopDownRuleInductionConfig(
        const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
    : minCoverage_(1), maxConditions_(0), maxHeadRefinements_(1), recalculatePredictions_(true),
      multiThreadingConfigPtr_(multiThreadingConfigPtr) {

}

uint32 TopDownRuleInductionConfig::getMinCoverage() const {
    return minCoverage_;
}

ITopDownRuleInductionConfig& TopDownRuleInductionConfig::setMinCoverage(uint32 minCoverage) {
    assertGreaterOrEqual<uint32>("minCoverage", minCoverage, 1);
    minCoverage_ = minCoverage;
    return *this;
}

uint32 TopDownRuleInductionConfig::getMaxConditions() const {
    return maxConditions_;
}

ITopDownRuleInductionConfig& TopDownRuleInductionConfig::setMaxConditions(uint32 maxConditions) {
    if (maxConditions != 0) { assertGreaterOrEqual<uint32>("maxConditions", maxConditions, 1); }
    maxConditions_ = maxConditions;
    return *this;
}

uint32 TopDownRuleInductionConfig::getMaxHeadRefinements() const {
    return maxHeadRefinements_;
}

ITopDownRuleInductionConfig& TopDownRuleInductionConfig::setMaxHeadRefinements(uint32 maxHeadRefinements) {
    if (maxHeadRefinements != 0) { assertGreaterOrEqual<uint32>("maxHeadRefinements", maxHeadRefinements, 1); }
    maxHeadRefinements_ = maxHeadRefinements;
    return *this;
}

bool TopDownRuleInductionConfig::getRecalculatePredictions() const {
    return recalculatePredictions_;
}

ITopDownRuleInductionConfig& TopDownRuleInductionConfig::setRecalculatePredictions(bool recalculatePredictions) {
    recalculatePredictions_ = recalculatePredictions;
    return *this;
}

std::unique_ptr<IRuleInductionFactory> TopDownRuleInductionConfig::createRuleInductionFactory(
        const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
    return std::make_unique<TopDownRuleInductionFactory>(minCoverage_, maxConditions_, maxHeadRefinements_,
                                                         recalculatePredictions_, numThreads);
}
