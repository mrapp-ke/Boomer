#include "common/rule_induction/rule_induction_top_down.hpp"
#include "common/rule_refinement/score_processor.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/validation.hpp"
#include "omp.h"
#include <unordered_map>


TopDownRuleInduction::TopDownRuleInduction(uint32 minCoverage, uint32 maxConditions, uint32 maxHeadRefinements,
                                           bool recalculatePredictions, uint32 numThreads)
    : minCoverage_(minCoverage), maxConditions_(maxConditions), maxHeadRefinements_(maxHeadRefinements),
      recalculatePredictions_(recalculatePredictions), numThreads_(numThreads) {
    assertGreaterOrEqual<uint32>("minCoverage", minCoverage, 1);
    if (maxConditions != 0) { assertGreaterOrEqual<uint32>("maxConditions", maxConditions, 1); }
    if (maxHeadRefinements != 0) { assertGreaterOrEqual<uint32>("maxHeadRefinements", maxHeadRefinements, 1); }
    assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
}

void TopDownRuleInduction::induceDefaultRule(IStatistics& statistics, IModelBuilder& modelBuilder) const {
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

bool TopDownRuleInduction::induceRule(IThresholds& thresholds, const IIndexVector& labelIndices,
                                      const IWeightVector& weights, IPartition& partition,
                                      IFeatureSampling& featureSampling, const IPruning& pruning,
                                      const IPostProcessor& postProcessor, RNG& rng,
                                      IModelBuilder& modelBuilder) const {
    // True, if the rule is learned on a sample of the available training examples, False otherwise
    bool instanceSamplingUsed = weights.hasZeroWeights();
    // The label indices for which the next refinement of the rule may predict
    const IIndexVector* currentLabelIndices = &labelIndices;
    // A (stack-allocated) list that contains the conditions in the rule's body (in the order they have been learned)
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

    // Search for the best refinement until no improvement in terms of the rule's quality score is possible anymore or
    // the maximum number of conditions has been reached...
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
            std::unique_ptr<IRuleRefinement>& ruleRefinementPtr = ruleRefinementsPtr->find(featureIndex)->second;
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

    if (bestHead == nullptr) {
        // No rule could be induced, because no useful condition could be found. This might be the case, if all examples
        // have the same values for the considered features.
        return false;
    } else {
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
                    coverageStatePtr.get() != nullptr ? *coverageStatePtr : thresholdsSubsetPtr->getCoverageState();
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
    }
}
