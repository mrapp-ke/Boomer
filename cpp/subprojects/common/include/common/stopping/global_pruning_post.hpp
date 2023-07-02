/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/macros.hpp"
#include "common/stopping/global_pruning.hpp"

/**
 * Defines an interface for all classes that allow to configure a stopping criterion that keeps track of the number of
 * rules in a model that perform best with respect to the examples in the training or holdout set according to a certain
 * measure.
 *
 * This stopping criterion assesses the performance of the current model after every `interval` rules and stores and
 * checks whether the current model is the best one evaluated so far.
 */
class MLRLCOMMON_API IPostPruningConfig {
    public:

        virtual ~IPostPruningConfig() {};

        /**
         * Returns whether the quality of the current model's predictions is measured on the holdout set, if available,
         * or if the training set is used instead.
         *
         * @return True, if the quality of the current model's predictions is measured on the holdout set, if available,
         *         false, if the training set is used instead
         */
        virtual bool isHoldoutSetUsed() const = 0;

        /**
         * Sets whether the quality of the current model's predictions should be measured on the holdout set, if
         * available, or if the training set should be used instead.
         *
         * @param useHoldoutSet True, if the quality of the current model's predictions should be measured on the
         *                      holdout set, if available, false, if the training set should be used instead
         * @return              A reference to an object of type `IPostPruningConfig` that allows further configuration
         *                      of the stopping criterion
         */
        virtual IPostPruningConfig& setUseHoldoutSet(bool useHoldoutSet) = 0;

        /**
         * Returns whether rules that have been induced, but are not used, should be removed from the final model or
         * not.
         *
         * @return True, if unused rules should be removed from the model, false otherwise
         */
        virtual bool isRemoveUnusedRules() const = 0;

        /**
         * Sets whether rules that have been induced, but are not used, should be removed from the final model or not.
         *
         * @param removeUnusedRules True, if unused rules should be removed from the model, false otherwise
         * @return                  A reference to an object of type `IPostPruningConfig` that allows further
         *                          configuration of the stopping criterion
         */
        virtual IPostPruningConfig& setRemoveUnusedRules(bool removeUnusedRules) = 0;

        /**
         * Returns the minimum number of rules that must be included in a model.
         *
         * @return The minimum number of rules that must be included in a model
         */
        virtual uint32 getMinRules() const = 0;

        /**
         * Sets the minimum number of rules that must be included in a model.
         *
         * @param minRules  The minimum number of rules that must be included in a model. Must be at least 1
         * @return          A reference to an object of type `IPostPruningConfig` that allows further configuration of
         *                  the stopping criterion
         */
        virtual IPostPruningConfig& setMinRules(uint32 minRules) = 0;

        /**
         * Returns the interval that is used to check whether the current model is the best one evaluated so far.
         *
         * @return The interval that is used to check whether the current model is the best one evaluated so far
         */
        virtual uint32 getInterval() const = 0;

        /**
         * Sets the interval that should be used to check whether the current model is the best one evaluated so far.
         *
         * @param interval  The interval that should be used to check whether the current model is the best one
         *                  evaluated so far, e.g., a value of 10 means that the best model may include 10, 20, ...
         *                  rules
         * @return          A reference to an object of type `IPostPruningConfig` that allows further configuration of
         *                  the stopping criterion
         */
        virtual IPostPruningConfig& setInterval(uint32 interval) = 0;
};

/**
 * Allows to configure a stopping criterion the keeps track of the number of rules in a model that perform best with
 * respect to the examples in the training or holdout set according to a certain measure.
 */
class PostPruningConfig final : public IGlobalPruningConfig,
                                public IPostPruningConfig {
    private:

        bool useHoldoutSet_;

        bool removeUnusedRules_;

        uint32 minRules_;

        uint32 interval_;

    public:

        PostPruningConfig();

        bool isHoldoutSetUsed() const override;

        IPostPruningConfig& setUseHoldoutSet(bool useHoldoutSet) override;

        bool isRemoveUnusedRules() const override;

        IPostPruningConfig& setRemoveUnusedRules(bool removeUnusedRules) override;

        uint32 getMinRules() const override;

        IPostPruningConfig& setMinRules(uint32 minRules) override;

        uint32 getInterval() const override;

        IPostPruningConfig& setInterval(uint32 interval) override;

        std::unique_ptr<IStoppingCriterionFactory> createStoppingCriterionFactory() const override;

        /**
         * @see `IGlobalPruningConfig::shouldUseHoldoutSet`
         */
        bool shouldUseHoldoutSet() const override;

        /**
         * @see `IGlobalPruningConfig::shouldRemoveUnusedRules`
         */
        bool shouldRemoveUnusedRules() const override;
};
