#include "common/rule_pruning/rule_pruning_no.hpp"

/**
 * An implementation of the class `IRulePruning` that does not actually perform any pruning.
 */
class NoRulePruning final : public IRulePruning {
    public:

        std::unique_ptr<ICoverageState> prune(IThresholdsSubset& thresholdsSubset, IPartition& partition,
                                              ConditionList& conditions,
                                              const AbstractPrediction& head) const override {
            return nullptr;
        }
};

/**
 * Allows to create instances of the type `IRulePruning` that do not actually perform any pruning.
 */
class NoRulePruningFactory final : public IRulePruningFactory {
    public:

        std::unique_ptr<IRulePruning> create() const override {
            return std::make_unique<NoRulePruning>();
        }
};

std::unique_ptr<IRulePruningFactory> NoRulePruningConfig::createRulePruningFactory() const {
    return std::make_unique<NoRulePruningFactory>();
}
