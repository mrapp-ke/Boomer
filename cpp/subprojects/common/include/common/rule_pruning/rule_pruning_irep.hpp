/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_pruning/rule_pruning.hpp"

/**
 * Allows to configure a strategy for pruning individual rules that prunes rules by following the principles of
 * "incremental reduced error pruning" (IREP).
 */
class IrepConfig final : public IRulePruningConfig {
    private:

        const RuleCompareFunction ruleCompareFunction_;

    public:

        /**
         * @param ruleCompareFunction An object of type `RuleCompareFunction` that defines the function that should be
         *                            used for comparing the quality of different rules
         */
        IrepConfig(RuleCompareFunction ruleCompareFunction);

        std::unique_ptr<IRulePruningFactory> createRulePruningFactory() const override;
};
