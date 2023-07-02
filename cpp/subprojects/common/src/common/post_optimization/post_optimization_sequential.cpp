#include "common/post_optimization/post_optimization_sequential.hpp"

#include "common/sampling/feature_sampling_predefined.hpp"
#include "common/util/validation.hpp"

#include <unordered_set>

/**
 * An implementation of the class `IModelBuilder` that allows to replace a single rule of an `IntermediateModelBuilder`.
 */
class RuleReplacementBuilder final : public IModelBuilder {
    private:

        IntermediateModelBuilder::IntermediateRule& intermediateRule_;

    public:

        /**
         * @param intermediateRule A reference to an object of type `IntermediateModelBuilder::IntermediateRule` that
         *                         should be replaced
         */
        RuleReplacementBuilder(IntermediateModelBuilder::IntermediateRule& intermediateRule)
            : intermediateRule_(intermediateRule) {}

        void setDefaultRule(std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) override {}

        void addRule(std::unique_ptr<ConditionList>& conditionListPtr,
                     std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) override {
            intermediateRule_.first = std::move(conditionListPtr);
            intermediateRule_.second = std::move(predictionPtr);
        }

        void setNumUsedRules(uint32 numUsedRules) override {}

        std::unique_ptr<IRuleModel> buildModel() override {
            return nullptr;
        }
};

/**
 * An implementation of the class `IPostOptimizationPhase` that optimizes each rule in a model by relearning it in the
 * context of the other rules.
 */
class SequentialPostOptimization final : public IPostOptimizationPhase {
    private:

        IntermediateModelBuilder& modelBuilder_;

        const uint32 numIterations_;

        const bool refineHeads_;

        const bool resampleFeatures_;

    public:

        /**
         * @param modelBuilder      A reference to an object of type `IntermediateModelBuilder` that provides access to
         *                          the existing rules
         * @param numIterations     The number of iterations to be performed. Must be at least 1
         * @param refineHeads       True, if the heads of rules should be refined when being relearned, false otherwise
         * @param resampleFeatures  True, if a new sample of the available features should be created when refining a
         *                          new rule, false otherwise
         */
        SequentialPostOptimization(IntermediateModelBuilder& modelBuilder, uint32 numIterations, bool refineHeads,
                                   bool resampleFeatures)
            : modelBuilder_(modelBuilder), numIterations_(numIterations), refineHeads_(refineHeads),
              resampleFeatures_(resampleFeatures) {}

        void optimizeModel(IThresholds& thresholds, const IRuleInduction& ruleInduction, IPartition& partition,
                           ILabelSampling& labelSampling, IInstanceSampling& instanceSampling,
                           IFeatureSampling& featureSampling, const IRulePruning& rulePruning,
                           const IPostProcessor& postProcessor, RNG& rng) const override {
            for (uint32 i = 0; i < numIterations_; i++) {
                for (auto it = modelBuilder_.begin(); it != modelBuilder_.end(); it++) {
                    IntermediateModelBuilder::IntermediateRule& intermediateRule = *it;
                    const ConditionList& conditionList = *intermediateRule.first;
                    const AbstractEvaluatedPrediction& prediction = *intermediateRule.second;

                    // Create a new subset of the given thresholds...
                    const IWeightVector& weights = instanceSampling.sample(rng);
                    std::unique_ptr<IThresholdsSubset> thresholdsSubsetPtr = weights.createThresholdsSubset(thresholds);

                    // Filter the thresholds subset according to the conditions of the current rule...
                    for (auto it2 = conditionList.cbegin(); it2 != conditionList.cend(); it2++) {
                        const Condition& condition = *it2;
                        thresholdsSubsetPtr->filterThresholds(condition);
                    }

                    // Revert the statistics based on the predictions of the current rule...
                    thresholdsSubsetPtr->revertPrediction(prediction);

                    // Learn a new rule...
                    const IIndexVector& labelIndices = refineHeads_ ? labelSampling.sample(rng) : prediction;
                    RuleReplacementBuilder ruleReplacementBuilder(intermediateRule);

                    if (resampleFeatures_) {
                        ruleInduction.induceRule(thresholds, labelIndices, weights, partition, featureSampling,
                                                 rulePruning, postProcessor, rng, ruleReplacementBuilder);
                    } else {
                        std::unordered_set<uint32> uniqueFeatureIndices;

                        for (auto it2 = conditionList.cbegin(); it2 != conditionList.cend(); it2++) {
                            const Condition& condition = *it2;
                            uniqueFeatureIndices.emplace(condition.featureIndex);
                        }

                        PartialIndexVector indexVector(uniqueFeatureIndices.size());
                        PartialIndexVector::iterator indexIterator = indexVector.begin();

                        for (auto it2 = uniqueFeatureIndices.cbegin(); it2 != uniqueFeatureIndices.cend(); it2++) {
                            *indexIterator = *it2;
                            indexIterator++;
                        }

                        PredefinedFeatureSampling predefinedFeatureSampling(indexVector);
                        ruleInduction.induceRule(thresholds, labelIndices, weights, partition,
                                                 predefinedFeatureSampling, rulePruning, postProcessor, rng,
                                                 ruleReplacementBuilder);
                    }
                }
            }
        }
};

/**
 * Allows to create instances of the type `IPostOptimizationPhase` that optimize each rule in a model by relearning it
 * in the context of the other rules.
 */
class SequentialPostOptimizationFactory final : public IPostOptimizationPhaseFactory {
    private:

        const uint32 numIterations_;

        const bool refineHeads_;

        const bool resampleFeatures_;

    public:

        /**
         * @param numIterations     The number of iterations to be performed. Must be at least 1
         * @param refineHeads       True, if the heads of rules should be refined when being relearned, false otherwise
         * @param resampleFeatures  True, if a new sample of the available features should be created when refining a
         *                          new rule, false otherwise
         */
        SequentialPostOptimizationFactory(uint32 numIterations, bool refineHeads, bool resampleFeatures)
            : numIterations_(numIterations), refineHeads_(refineHeads), resampleFeatures_(resampleFeatures) {}

        std::unique_ptr<IPostOptimizationPhase> create(IntermediateModelBuilder& modelBuilder) const override {
            return std::make_unique<SequentialPostOptimization>(modelBuilder, numIterations_, refineHeads_,
                                                                resampleFeatures_);
        }
};

SequentialPostOptimizationConfig::SequentialPostOptimizationConfig()
    : numIterations_(2), refineHeads_(false), resampleFeatures_(true) {}

uint32 SequentialPostOptimizationConfig::getNumIterations() const {
    return numIterations_;
}

ISequentialPostOptimizationConfig& SequentialPostOptimizationConfig::setNumIterations(uint32 numIterations) {
    assertGreaterOrEqual<uint32>("numIterations", numIterations, 1);
    numIterations_ = numIterations;
    return *this;
}

bool SequentialPostOptimizationConfig::areHeadsRefined() const {
    return refineHeads_;
}

ISequentialPostOptimizationConfig& SequentialPostOptimizationConfig::setRefineHeads(bool refineHeads) {
    refineHeads_ = refineHeads;
    return *this;
}

bool SequentialPostOptimizationConfig::areFeaturesResampled() const {
    return resampleFeatures_;
}

ISequentialPostOptimizationConfig& SequentialPostOptimizationConfig::setResampleFeatures(bool resampleFeatures) {
    resampleFeatures_ = resampleFeatures;
    return *this;
}

std::unique_ptr<IPostOptimizationPhaseFactory> SequentialPostOptimizationConfig::createPostOptimizationPhaseFactory()
  const {
    return std::make_unique<SequentialPostOptimizationFactory>(numIterations_, refineHeads_, resampleFeatures_);
}
