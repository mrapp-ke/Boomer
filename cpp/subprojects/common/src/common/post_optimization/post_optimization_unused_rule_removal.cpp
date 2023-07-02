#include "common/post_optimization/post_optimization_unused_rule_removal.hpp"

/**
 * An implementation of the class `IPostOptimizationPhase` that removes unused rules from a model.
 */
class UnusedRuleRemoval final : public IPostOptimizationPhase {
    private:

        IntermediateModelBuilder& modelBuilder_;

    public:

        /**
         * @param modelBuilder A reference to an object of type `IntermediateModelBuilder` that provides access to the
         *                     rules in a model
         */
        UnusedRuleRemoval(IntermediateModelBuilder& modelBuilder) : modelBuilder_(modelBuilder) {}

        void optimizeModel(IThresholds& thresholds, const IRuleInduction& ruleInduction, IPartition& partition,
                           ILabelSampling& labelSampling, IInstanceSampling& instanceSampling,
                           IFeatureSampling& featureSampling, const IRulePruning& rulePruning,
                           const IPostProcessor& postProcessor, RNG& rng) const override {
            uint32 numUsedRules = modelBuilder_.getNumUsedRules();

            if (numUsedRules > 0) {
                while (modelBuilder_.getNumRules() > numUsedRules) {
                    modelBuilder_.removeLastRule();
                }

                modelBuilder_.setNumUsedRules(0);
            }
        }
};

/**
 * Allows to create instances of the type `IPostOptimizationPhase` that remove unused rules from a model.
 */
class UnusedRuleRemovalFactory final : public IPostOptimizationPhaseFactory {
    public:

        std::unique_ptr<IPostOptimizationPhase> create(IntermediateModelBuilder& modelBuilder) const override {
            return std::make_unique<UnusedRuleRemoval>(modelBuilder);
        }
};

std::unique_ptr<IPostOptimizationPhaseFactory> UnusedRuleRemovalConfig::createPostOptimizationPhaseFactory() const {
    return std::make_unique<UnusedRuleRemovalFactory>();
}
