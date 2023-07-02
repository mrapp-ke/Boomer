/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/macros.hpp"
#include "common/stopping/aggregation_function.hpp"
#include "common/stopping/global_pruning.hpp"

/**
 * Defines an interface for all classes that allow to configure a stopping criterion that stops the induction of rules
 * as soon as the quality of a model's predictions for the examples in the training or holdout set do not improve
 * according to a certain measure.
 *
 * This stopping criterion assesses the performance of the current model after every `updateInterval` rules and stores
 * its quality in a buffer that keeps track of the last `numCurrent` iterations. If the capacity of this buffer is
 * already reached, the oldest quality is passed to a buffer of size `numPast`. Every `stopInterval` rules, it is
 * decided whether the rule induction should be stopped. For this reason, the `numCurrent` qualities in the first
 * buffer, as well as the `numPast` qualities in the second buffer are aggregated according to a certain
 * `aggregationFunction`. If the percentage improvement, which results from comparing the more recent qualities from the
 * first buffer to the older qualities from the second buffer, is greater than a certain `minImprovement`, the rule
 * induction is continued, otherwise it is stopped.
 */
class MLRLCOMMON_API IPrePruningConfig {
    public:

        virtual ~IPrePruningConfig() {};

        /**
         * Returns the type of the aggregation function that is used to aggregate the values that are stored in a
         * buffer.
         *
         * @return A value of the enum `AggregationFunction` that specifies the type of the aggregation function that is
         *         used to aggregate the values that are stored in a buffer
         */
        virtual AggregationFunction getAggregationFunction() const = 0;

        /**
         * Sets the type of the aggregation function that should be used to aggregate the values that are stored in a
         * buffer.
         *
         * @param aggregationFunction   A value of the enum `AggregationFunction` that specifies the type of the
         *                              aggregation function that should be used to aggregate the values that are stored
         *                              in a buffer
         * @return                      A reference to an object of type `IPrePruningConfig` that allows further
         *                              configuration of the stopping criterion
         */
        virtual IPrePruningConfig& setAggregationFunction(AggregationFunction aggregationFunction) = 0;

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
         * @return              A reference to an object of type `IPrePruningConfig` that allows further configuration
         *                      of the stopping criterion
         */
        virtual IPrePruningConfig& setUseHoldoutSet(bool useHoldoutSet) = 0;

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
         * @return                  A reference to an object of type `IPrePruningConfig` that allows further
         *                          configuration of the stopping criterion
         */
        virtual IPrePruningConfig& setRemoveUnusedRules(bool removeUnusedRules) = 0;

        /**
         * Returns the minimum number of rules that must have been learned until the induction of rules might be
         * stopped.
         *
         * @return The minimum number of rules that must have been learned until the induction of rules might be stopped
         */
        virtual uint32 getMinRules() const = 0;

        /**
         * Sets the minimum number of rules that must have been learned until the induction of rules might be stopped.
         *
         * @param minRules  The minimum number of rules that must have been learned until the induction of rules might
         *                  be stopped. Must be at least 1
         * @return          A reference to an object of type `IPrePruningConfig` that allows further configuration of
         *                  the stopping criterion
         */
        virtual IPrePruningConfig& setMinRules(uint32 minRules) = 0;

        /**
         * Returns the interval that is used to update the quality of the current model.
         *
         * @return The interval that is used to update the quality of the current model
         */
        virtual uint32 getUpdateInterval() const = 0;

        /**
         * Sets the interval that should be used to update the quality of the current model.
         *
         * @param updateInterval    The interval that should be used to update the quality of the current model, e.g., a
         *                          value of 5 means that the model quality is assessed every 5 rules. Must be at least
         *                          1
         * @return                  A reference to an object of type `IPrePruningConfig` that allows further
         *                          configuration of the stopping criterion
         */
        virtual IPrePruningConfig& setUpdateInterval(uint32 updateInterval) = 0;

        /**
         * Returns the interval that is used to decide whether the induction of rules should be stopped.
         *
         * @return The interval that is used to decide whether the induction of rules should be stopped
         */
        virtual uint32 getStopInterval() const = 0;

        /**
         * Sets the interval that should be used to decide whether the induction of rules should be stopped.
         *
         * @param stopInterval  The interval that should be used to decide whether the induction of rules should be
         *                      stopped, e.g., a value of 10 means that the rule induction might be stopped after 10,
         *                      20, ... rules. Must be a multiple of the update interval
         * @return              A reference to an object of type `IPrePruningConfig` that allows further configuration
         *                      of the stopping criterion
         */
        virtual IPrePruningConfig& setStopInterval(uint32 stopInterval) = 0;

        /**
         * Returns the number of quality stores of past iterations that are stored in a buffer.
         *
         * @return The number of quality stores of past iterations that are stored in a buffer
         */
        virtual uint32 getNumPast() const = 0;

        /**
         * Sets the number of past iterations that should be stored in a buffer.
         *
         * @param numPast   The number of past iterations that should be be stored in a buffer. Must be at least 1
         * @return          A reference to an object of type `IPrePruningConfig` that allows further configuration of
         *                  the stopping criterion
         */
        virtual IPrePruningConfig& setNumPast(uint32 numPast) = 0;

        /**
         * Returns the number of the most recent iterations that are stored in a buffer.
         *
         * @return The number of the most recent iterations that are stored in a buffer
         */
        virtual uint32 getNumCurrent() const = 0;

        /**
         * Sets the number of the most recent iterations that should be stored in a buffer.
         *
         * @param numCurrent    The number of the most recent iterations that should be stored in a buffer. Must be at
         *                      least 1
         * @return              A reference to an object of type `IPrePruningConfig` that allows further configuration
         *                      of the stopping criterion
         */
        virtual IPrePruningConfig& setNumCurrent(uint32 numCurrent) = 0;

        /**
         * Returns the minimum improvement that must be reached for the rule induction to be continued.
         *
         * @return The minimum improvement that must be reached for the rule induction to be continued
         */
        virtual float64 getMinImprovement() const = 0;

        /**
         * Sets the minimum improvement that must be reached for the rule induction to be continued.
         *
         * @param minImprovement    The minimum improvement in percent that must be reached for the rule induction to be
         *                          continued. Must be in [0, 1]
         * @return                  A reference to an object of type `IPrePruningConfig` that allows further
         *                          configuration of the stopping criterion
         */
        virtual IPrePruningConfig& setMinImprovement(float64 minImprovement) = 0;
};

/**
 * Allows to configure a stopping criterion that stops the induction of rules as soon as the quality of a model's
 * predictions for the examples in the training or holdout set do not improve according to a certain measure.
 */
class PrePruningConfig final : public IGlobalPruningConfig,
                               public IPrePruningConfig {
    private:

        AggregationFunction aggregationFunction_;

        bool useHoldoutSet_;

        bool removeUnusedRules_;

        uint32 minRules_;

        uint32 updateInterval_;

        uint32 stopInterval_;

        uint32 numPast_;

        uint32 numCurrent_;

        float64 minImprovement_;

    public:

        PrePruningConfig();

        AggregationFunction getAggregationFunction() const override;

        IPrePruningConfig& setAggregationFunction(AggregationFunction aggregationFunction) override;

        bool isHoldoutSetUsed() const override;

        IPrePruningConfig& setUseHoldoutSet(bool useHoldoutSet) override;

        bool isRemoveUnusedRules() const override;

        IPrePruningConfig& setRemoveUnusedRules(bool removeUnusedRules) override;

        uint32 getMinRules() const override;

        IPrePruningConfig& setMinRules(uint32 minRules) override;

        uint32 getUpdateInterval() const override;

        IPrePruningConfig& setUpdateInterval(uint32 updateInterval) override;

        uint32 getStopInterval() const override;

        IPrePruningConfig& setStopInterval(uint32 stopInterval) override;

        uint32 getNumPast() const override;

        IPrePruningConfig& setNumPast(uint32 numPast) override;

        uint32 getNumCurrent() const override;

        IPrePruningConfig& setNumCurrent(uint32 numCurrent) override;

        float64 getMinImprovement() const override;

        IPrePruningConfig& setMinImprovement(float64 minImprovement) override;

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
