/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_induction/rule_model_assemblage.hpp"
#include "common/macros.hpp"


/**
 * Defines an interface for all classes that allow to configure an algorithm that sequentially induces several rules,
 * optionally starting with a default rule, that are added to a rule-based model.
 */
class MLRLCOMMON_API ISequentialRuleModelAssemblageConfig {

    public:

        virtual ~ISequentialRuleModelAssemblageConfig() { };

        /**
         * Returns whether a default rule should be used or not.
         *
         * @return True, if a default rule should be used, false otherwise
         */
        virtual bool getUseDefaultRule() const = 0;

        /**
         * Sets whether a default rule should be used or not.
         *
         * @param useDefaultRule    True, if a default rule should be used, false otherwise
         * @return                  A reference to an object of type `SequentialRuleModelAssemblageConfig` that allows
         *                          further configuration of the algorithm for the induction of several rules that are
         *                          added to a rule-based model
         */
        virtual ISequentialRuleModelAssemblageConfig& setUseDefaultRule(bool useDefaultRule) = 0;

};

/**
 * Allows to configure an algorithm that sequentially induces several rules, optionally starting with a default rule,
 * that are added to a rule-based model.
 */
class SequentialRuleModelAssemblageConfig final : public IRuleModelAssemblageConfig,
                                                  public ISequentialRuleModelAssemblageConfig {

    private:

        bool useDefaultRule_;

    public:

        SequentialRuleModelAssemblageConfig();

        bool getUseDefaultRule() const override;

        ISequentialRuleModelAssemblageConfig& setUseDefaultRule(bool useDefaultRule) override;

        std::unique_ptr<IRuleModelAssemblageFactory> createRuleModelAssemblageFactory() const override;

};
