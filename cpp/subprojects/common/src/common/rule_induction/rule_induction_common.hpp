/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/indices/index_vector_complete.hpp"
#include "common/rule_induction/rule_induction.hpp"
#include "common/rule_refinement/score_processor.hpp"

/**
 * An abstract base class for all classes that implement an algorithm for the induction of individual rules.
 */
class AbstractRuleInduction : public IRuleInduction {
    private:

        const bool recalculatePredictions_;

    protected:

        /**
         * Must be implemented by subclasses in order to grow a rule.
         *
         * @param thresholds        A reference to an object of type `IThresholds` that provides access to the
         *                          thresholds that may be used by the conditions of the rule
         * @param labelIndices      A reference to an object of type `IIndexVector` that provides access to the indices
         *                          of the labels for which the rule may predict
         * @param weights           A reference to an object of type `IWeightVector` that provides access to the weights
         *                          of individual training examples
         * @param partition         A reference to an object of type `IPartition` that provides access to the indices of
         *                          the training examples that belong to the training set and the holdout set,
         *                          respectively
         * @param featureSampling   A reference to an object of type `IFeatureSampling` that should be used for sampling
         *                          the features that may be used by a new condition
         * @param rng               A reference to an object of type `RNG` that implements the random number generator
         *                          to be used
         * @param conditionListPtr  A reference to an unique pointer of type `ConditionList` that should be used to
         *                          store the conditions of the rule
         * @param headPtr           A reference to an unique pointer of type `AbstractEvaluatedPrediction` that should
         *                          be used to store the head of the rule
         * @return                  An unique pointer to an object of type `IThresholdsSubset` that has been used to
         *                          grow the rule
         */
        virtual std::unique_ptr<IThresholdsSubset> growRule(
          IThresholds& thresholds, const IIndexVector& labelIndices, const IWeightVector& weights,
          IPartition& partition, IFeatureSampling& featureSampling, RNG& rng,
          std::unique_ptr<ConditionList>& conditionListPtr,
          std::unique_ptr<AbstractEvaluatedPrediction>& headPtr) const = 0;

    public:

        /**
         * @param recalculatePredictions True, if the predictions of rules should be recalculated on all training
         *                               examples, if some of the examples have zero weights, false otherwise
         */
        AbstractRuleInduction(bool recalculatePredictions) : recalculatePredictions_(recalculatePredictions) {}

        virtual ~AbstractRuleInduction() override {};

        void induceDefaultRule(IStatistics& statistics, IModelBuilder& modelBuilder) const override final {
            uint32 numStatistics = statistics.getNumStatistics();
            uint32 numLabels = statistics.getNumLabels();
            CompleteIndexVector labelIndices(numLabels);
            EqualWeightVector weights(numStatistics);
            std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = statistics.createSubset(labelIndices, weights);

            for (uint32 i = 0; i < numStatistics; i++) {
                statisticsSubsetPtr->addToSubset(i);
            }

            const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScores();
            std::unique_ptr<AbstractEvaluatedPrediction> defaultPredictionPtr;
            ScoreProcessor scoreProcessor(defaultPredictionPtr);
            scoreProcessor.processScores(scoreVector);

            for (uint32 i = 0; i < numStatistics; i++) {
                defaultPredictionPtr->apply(statistics, i);
            }

            modelBuilder.setDefaultRule(defaultPredictionPtr);
        }

        bool induceRule(IThresholds& thresholds, const IIndexVector& labelIndices, const IWeightVector& weights,
                        IPartition& partition, IFeatureSampling& featureSampling, const IRulePruning& rulePruning,
                        const IPostProcessor& postProcessor, RNG& rng,
                        IModelBuilder& modelBuilder) const override final {
            std::unique_ptr<ConditionList> conditionListPtr;
            std::unique_ptr<AbstractEvaluatedPrediction> headPtr;
            std::unique_ptr<IThresholdsSubset> thresholdsSubsetPtr = this->growRule(
              thresholds, labelIndices, weights, partition, featureSampling, rng, conditionListPtr, headPtr);

            if (headPtr) {
                if (weights.hasZeroWeights()) {
                    // Prune rule...
                    IStatisticsProvider& statisticsProvider = thresholds.getStatisticsProvider();
                    statisticsProvider.switchToPruningRuleEvaluation();
                    std::unique_ptr<ICoverageState> coverageStatePtr =
                      rulePruning.prune(*thresholdsSubsetPtr, partition, *conditionListPtr, *headPtr);
                    statisticsProvider.switchToRegularRuleEvaluation();

                    // Re-calculate the scores in the head based on the entire training data...
                    if (recalculatePredictions_) {
                        const ICoverageState& coverageState =
                          coverageStatePtr ? *coverageStatePtr : thresholdsSubsetPtr->getCoverageState();
                        partition.recalculatePrediction(*thresholdsSubsetPtr, coverageState, *headPtr);
                    }
                }

                // Apply post-processor...
                postProcessor.postProcess(*headPtr);

                // Update the statistics by applying the predictions of the new rule...
                thresholdsSubsetPtr->applyPrediction(*headPtr);

                // Add the induced rule to the model...
                modelBuilder.addRule(conditionListPtr, headPtr);
                return true;
            } else {
                // No rule could be induced, because no useful condition could be found. This might be the case, if all
                // examples have the same values for the considered features.
                return false;
            }
        }
};
