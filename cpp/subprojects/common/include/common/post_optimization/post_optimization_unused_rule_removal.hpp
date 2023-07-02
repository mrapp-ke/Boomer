/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/post_optimization/post_optimization.hpp"

/**
 * Allows to configure a method that removes unused rules from a model.
 */
class UnusedRuleRemovalConfig final : public IPostOptimizationPhaseConfig {
    public:

        std::unique_ptr<IPostOptimizationPhaseFactory> createPostOptimizationPhaseFactory() const override;
};
