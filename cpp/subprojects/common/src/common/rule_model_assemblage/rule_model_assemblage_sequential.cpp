#include "common/rule_model_assemblage/rule_model_assemblage_sequential.hpp"

/**
 * Allows to sequentially induce several rules, optionally starting with a default rule, that are added to a rule-based
 * model.
 */
class SequentialRuleModelAssemblage final : public IRuleModelAssemblage {
    private:

        const std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr_;

        const bool useDefaultRule_;

    public:

        /**
         * @param stoppingCriterionFactoryPtr   An unique pointer to an object of type `IStoppingCriterionFactory` that
         *                                      allows to create the implementations to be used to decide whether
         *                                      additional rules should be induced or not
         * @param useDefaultRule                True, if a default rule should be used, False otherwise
         */
        SequentialRuleModelAssemblage(std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr,
                                      bool useDefaultRule)
            : stoppingCriterionFactoryPtr_(std::move(stoppingCriterionFactoryPtr)), useDefaultRule_(useDefaultRule) {}

        void induceRules(const IRuleInduction& ruleInduction, const IRulePruning& rulePruning,
                         const IPostProcessor& postProcessor, IPartition& partition, ILabelSampling& labelSampling,
                         IInstanceSampling& instanceSampling, IFeatureSampling& featureSampling,
                         IStatisticsProvider& statisticsProvider, IThresholds& thresholds, IModelBuilder& modelBuilder,
                         RNG& rng) const override {
            uint32 numRules = useDefaultRule_ ? 1 : 0;
            uint32 numUsedRules = 0;

            // Induce default rule, if necessary...
            if (useDefaultRule_) {
                ruleInduction.induceDefaultRule(statisticsProvider.get(), modelBuilder);
            }

            statisticsProvider.switchToRegularRuleEvaluation();

            // Induce the remaining rules...
            std::unique_ptr<IStoppingCriterion> stoppingCriterionPtr =
              partition.createStoppingCriterion(*stoppingCriterionFactoryPtr_);

            while (true) {
                IStoppingCriterion::Result stoppingCriterionResult =
                  stoppingCriterionPtr->test(statisticsProvider.get(), numRules);

                if (stoppingCriterionResult.numUsedRules != 0) {
                    numUsedRules = stoppingCriterionResult.numUsedRules;
                }

                if (stoppingCriterionResult.stop) {
                    break;
                }

                const IWeightVector& weights = instanceSampling.sample(rng);
                const IIndexVector& labelIndices = labelSampling.sample(rng);
                bool success = ruleInduction.induceRule(thresholds, labelIndices, weights, partition, featureSampling,
                                                        rulePruning, postProcessor, rng, modelBuilder);

                if (success) {
                    numRules++;
                } else {
                    break;
                }
            }

            // Set the number of used rules...
            modelBuilder.setNumUsedRules(numUsedRules);
        }
};

/**
 * A factory that allows to create instances of the class `IRuleModelAssemblage` that allow to sequentially induce
 * several rules, optionally starting with a default rule, that are added to a rule-based model.
 */
class SequentialRuleModelAssemblageFactory final : public IRuleModelAssemblageFactory {
    private:

        const bool useDefaultRule_;

    public:

        /**
         * @param useDefaultRule True, if a default rule should be used, false otherwise
         */
        SequentialRuleModelAssemblageFactory(bool useDefaultRule) : useDefaultRule_(useDefaultRule) {}

        std::unique_ptr<IRuleModelAssemblage> create(
          std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr) const override {
            return std::make_unique<SequentialRuleModelAssemblage>(std::move(stoppingCriterionFactoryPtr),
                                                                   useDefaultRule_);
        }
};

SequentialRuleModelAssemblageConfig::SequentialRuleModelAssemblageConfig(
  const std::unique_ptr<IDefaultRuleConfig>& defaultRuleConfigPtr)
    : defaultRuleConfigPtr_(defaultRuleConfigPtr) {}

std::unique_ptr<IRuleModelAssemblageFactory> SequentialRuleModelAssemblageConfig::createRuleModelAssemblageFactory(
  const IRowWiseLabelMatrix& labelMatrix) const {
    bool useDefaultRule = defaultRuleConfigPtr_->isDefaultRuleUsed(labelMatrix);
    return std::make_unique<SequentialRuleModelAssemblageFactory>(useDefaultRule);
}
