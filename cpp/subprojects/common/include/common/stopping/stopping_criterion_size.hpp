/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"
#include "common/macros.hpp"


/**
 * Defines an interface for all classes that allow to configure a stopping criterion that ensures that the number of
 * induced rules does not exceed a certain maximum.
 */
class MLRLCOMMON_API ISizeStoppingCriterionConfig {

    public:

        virtual ~ISizeStoppingCriterionConfig() { };

        /**
         * Returns the maximum number of rules that are induced.
         *
         * @return The maximum number of rules that are induced
         */
        virtual uint32 getMaxRules() const = 0;

        /**
         * Sets the maximum number of rules that should be induced.
         *
         * @param maxRules  The maximum number of rules that should be induced. Must be at least 1
         * @return          A reference to an object of type `ISizeStoppingCriterionConfig` that allows further
         *                  configuration of the stopping criterion
         */
        virtual ISizeStoppingCriterionConfig& setMaxRules(uint32 maxRules) = 0;

};

/**
 * Allows to configure a stopping criterion that ensures that the number of induced rules does not exceed a certain
 * maximum.
 */
class SizeStoppingCriterionConfig final : public IStoppingCriterionConfig, public ISizeStoppingCriterionConfig {

    private:

        uint32 maxRules_;

    public:

        SizeStoppingCriterionConfig();

        uint32 getMaxRules() const override;

        ISizeStoppingCriterionConfig& setMaxRules(uint32 maxRules) override;

        std::unique_ptr<IStoppingCriterionFactory> createStoppingCriterionFactory() const override;

};
