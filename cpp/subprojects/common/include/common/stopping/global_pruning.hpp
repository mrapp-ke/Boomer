/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"

/**
 * Defines an interface for all classes that allow to configure a stopping criterion that allows to decide how many
 * rules should be included in a model, such that its performance is optimized globally.
 */
class IGlobalPruningConfig : public IStoppingCriterionConfig {
    public:

        virtual ~IGlobalPruningConfig() override {};

        /**
         * Returns whether a holdout set should be used, if available, or not.
         *
         * @return True, if a holdout set should be used, false otherwise
         */
        virtual bool shouldUseHoldoutSet() const = 0;

        /**
         * Returns whether unused rules should be removed from the final model or not.
         *
         * @return True, if unused rules should be removed, false otherwise
         */
        virtual bool shouldRemoveUnusedRules() const = 0;
};
