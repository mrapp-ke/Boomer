/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"
#include "common/macros.hpp"


/**
 * Defines an interface for all classes that allow to configure a stopping criterion that stops the induction of rules
 * as soon as the quality of a model's predictions for the examples in a holdout set do not improve according to a
 * certain measure.
 *
 * This stopping criterion assesses the performance of the current model after every `updateInterval` rules and stores
 * the resulting quality score in a buffer that keeps track of the last `numCurrent` scores. If the capacity of this
 * buffer is already reached, the oldest score is passed to a buffer of size `numPast`. Every `stopInterval` rules, it
 * is decided whether the rule induction should be stopped. For this reason, the `numCurrent` scores in the first
 * buffer, as well as the `numPast` scores in the second buffer are aggregated according to a certain
 * `aggregationFunction`. If the percentage improvement, which results from comparing the more recent scores from the
 * first buffer to the older scores from the second buffer, is greater than a certain `minImprovement`, the rule
 * induction is continued, otherwise it is stopped.
 */
class MLRLCOMMON_API IMeasureStoppingCriterionConfig {

    public:

        /**
         * Specifies different types of aggregation functions that allow to aggregate the values that are stored in a
         * buffer.
         */
        enum AggregationFunction : uint8 {

            /**
             * An aggregation function that finds the minimum value in a buffer.
             */
            MIN = 0,

            /**
             * An aggregation function that finds the maximum value in a buffer.
             */
            MAX = 1,

            /**
             * An aggregation function that calculates the arithmetic mean of the values in a buffer.
             */
            ARITHMETIC_MEAN = 2

        };

        virtual ~IMeasureStoppingCriterionConfig() { };

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
         * @return                      A reference to an object of type `MeasureStoppingCriterionConfig` that allows
         *                              further configuration of the stopping criterion
         */
        virtual IMeasureStoppingCriterionConfig& setAggregationFunction(AggregationFunction aggregationFunction) = 0;

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
         * @return          A reference to an object of type `MeasureStoppingCriterionConfig` that allows further
         *                  configuration of the stopping criterion
         */
        virtual IMeasureStoppingCriterionConfig& setMinRules(uint32 minRules) = 0;

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
         * @return                  A reference to an object of type `MeasureStoppingCriterionConfig` that allows
         *                          further configuration of the stopping criterion
         */
        virtual IMeasureStoppingCriterionConfig& setUpdateInterval(uint32 updateInterval) = 0;

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
         * @return              A reference to an object of type `MeasureStoppingCriterionConfig` that allows further
         *                      configuration of the stopping criterion
         */
        virtual IMeasureStoppingCriterionConfig& setStopInterval(uint32 stopInterval) = 0;

        /**
         * Returns the number of quality stores of past iterations that are stored in a buffer.
         *
         * @return The number of quality stores of past iterations that are stored in a buffer
         */
        virtual uint32 getNumPast() const = 0;

        /**
         * Sets the number of quality scores of past iterations that should be stored in a buffer.
         *
         * @param numPast   The number of quality scores of past iterations that should be be stored in a buffer. Must
         *                  be at least 1
         * @return          A reference to an object of type `MeasureStoppingCriterionConfig` that allows further
         *                  configuration of the stopping criterion
         */
        virtual IMeasureStoppingCriterionConfig& setNumPast(uint32 numPast) = 0;

        /**
         * Returns the number of quality scores of the most recent iterations that are stored in a buffer.
         *
         * @return The number of quality scores of the most recent iterations that are stored in a buffer
         */
        virtual uint32 getNumCurrent() const = 0;

        /**
         * Sets the number of quality scores of the most recent iterations that should be stored in a buffer.
         *
         * @param numCurrent    The number of quality scores of the most recent iterations that should be stored in a
         *                      buffer. Must be at least 1
         * @return              A reference to an object of type `MeasureStoppingCriterionConfig` that allows further
         *                      configuration of the stopping criterion
         */
        virtual IMeasureStoppingCriterionConfig& setNumCurrent(uint32 numCurrent) = 0;

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
         * @return                  A reference to an object of type `MeasureStoppingCriterionConfig` that allows
         *                          further configuration of the stopping criterion
         */
        virtual IMeasureStoppingCriterionConfig& setMinImprovement(float64 minImprovement) = 0;

        /**
         * Returns whether the induction of rules is forced to be stopped, if the stopping criterion is met.
         *
         * @return True, if the induction of rules is forced to be stopped, if the stopping criterion is met, false, if
         *         only the time of stopping is stored
         */
        virtual bool getForceStop() const = 0;

        /**
         * Sets whether the induction of rules should be forced to be stopped, if the stopping criterion is met.
         *
         * @param forceStop True, if the induction of rules should be forced to be stopped, if the stopping criterion is
         *                  met, false, if only the time of stopping should be stored
         * @return          A reference to an object of type `MeasureStoppingCriterionConfig` that allows further
         *                  configuration of the stopping criterion
         */
        virtual IMeasureStoppingCriterionConfig& setForceStop(bool forceStop) = 0;

};

/**
 * Allows to configure a stopping criterion that stops the induction of rules as soon as the quality of a model's
 * predictions for the examples in a holdout set do not improve according to a certain measure.
 */
class MeasureStoppingCriterionConfig final : public IStoppingCriterionConfig, public IMeasureStoppingCriterionConfig {

    private:

        AggregationFunction aggregationFunction_;

        uint32 minRules_;

        uint32 updateInterval_;

        uint32 stopInterval_;

        uint32 numPast_;

        uint32 numCurrent_;

        float64 minImprovement_;

        bool forceStop_;

    public:

        MeasureStoppingCriterionConfig();

        AggregationFunction getAggregationFunction() const override;

        IMeasureStoppingCriterionConfig& setAggregationFunction(AggregationFunction aggregationFunction) override;

        uint32 getMinRules() const override;

        IMeasureStoppingCriterionConfig& setMinRules(uint32 minRules) override;

        uint32 getUpdateInterval() const override;

        IMeasureStoppingCriterionConfig& setUpdateInterval(uint32 updateInterval) override;

        uint32 getStopInterval() const override;

        IMeasureStoppingCriterionConfig& setStopInterval(uint32 stopInterval) override;

        uint32 getNumPast() const override;

        IMeasureStoppingCriterionConfig& setNumPast(uint32 numPast) override;

        uint32 getNumCurrent() const override;

        IMeasureStoppingCriterionConfig& setNumCurrent(uint32 numCurrent) override;

        float64 getMinImprovement() const override;

        IMeasureStoppingCriterionConfig& setMinImprovement(float64 minImprovement) override;

        bool getForceStop() const override;

        IMeasureStoppingCriterionConfig& setForceStop(bool forceStop) override;

        std::unique_ptr<IStoppingCriterionFactory> createStoppingCriterionFactory() const override;

};
