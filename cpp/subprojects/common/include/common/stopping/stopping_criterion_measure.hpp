/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"
#include "common/measures/measure_evaluation.hpp"
#include "common/data/ring_buffer.hpp"
#include <memory>


/**
 * Defines an interface for all classes that allow to aggregate the values that are stored in a buffer.
 */
class IAggregationFunction {

    public:

        virtual ~IAggregationFunction() { };

        /**
         * Aggregates the values that are stored in a buffer.
         *
         * @param begin An iterator to the beginning of the buffer
         * @param end   An iterator to the end of the buffer
         * @return      The aggregated value
         */
        virtual float64 aggregate(RingBuffer<float64>::const_iterator begin,
                                  RingBuffer<float64>::const_iterator end) const = 0;

};

/**
 * Allows to aggregate the values that are stored in a buffer by finding the minimum value.
 */
class MinFunction : public IAggregationFunction {

    public:

        float64 aggregate(RingBuffer<float64>::const_iterator begin,
                          RingBuffer<float64>::const_iterator end) const override;

};

/**
 * Allows to aggregate the values that are stored in a buffer by finding the maximum value.
 */
class MaxFunction : public IAggregationFunction {

    public:

        float64 aggregate(RingBuffer<float64>::const_iterator begin,
                          RingBuffer<float64>::const_iterator end) const override;

};

/**
 * Allows to aggregate the values that are stored in a buffer by calculating the arithmetic mean.
 */
class ArithmeticMeanFunction : public IAggregationFunction {

    public:

        float64 aggregate(RingBuffer<float64>::const_iterator begin,
                          RingBuffer<float64>::const_iterator end) const override;

};

/**
 * A stopping criterion that stops the induction of rules as soon as the quality of a model's predictions for the
 * examples in a holdout set do not improve according a certain measure.
 *
 * This stopping criterion assesses the performance of the current model after every `updateInterval` rules and stores
 * the resulting quality score in a buffer that keeps track of the last `numRecent` scores. If the capacity of this
 * buffer is already reached, the oldest score is passed to a buffer of size `numPast`. Every `stopInterval` rules, it
 * is decided whether the rule induction should be stopped. For this reason, the `numRecent` scores in the first buffer,
 * as well as the `numPast` scores in the second buffer are aggregated according to a certain `aggregationFunction`. If
 * the percentage improvement, which results from comparing the more recent scores from the first buffer to the older
 * scores from the second buffer, is greater than a certain `minImprovement`, the rule induction is continued,
 * otherwise it is stopped.
 */
class MeasureStoppingCriterion final : public IStoppingCriterion {

    private:

        std::unique_ptr<IEvaluationMeasure> measurePtr_;

        std::unique_ptr<IAggregationFunction> aggregationFunctionPtr_;

        uint32 updateInterval_;

        uint32 stopInterval_;

        float64 minImprovement_;

        RingBuffer<float64> pastBuffer_;

        RingBuffer<float64> recentBuffer_;

        uint32 offset_;

        Action stoppingAction_;

        float64 bestScore_;

        uint32 bestNumRules_;

        bool stopped_;

    public:

        /**
         * @param measurePtr                An unique pointer to an object of type `IEvaluationMeasure` that should be
         *                                  used to assess the quality of a model
         * @param aggregationFunctionPtr    An unique pointer to an object of type `IAggregationFunction` that should be
         *                                  used to aggregate the scores in the buffer
         * @param minRules                  The minimum number of rules that must have been learned until the induction
         *                                  of rules might be stopped. Must be at least 1
         * @param updateInterval            The interval to be used to update the quality of the current model, e.g., a
         *                                  value of 5 means that the model quality is assessed every 5 rules. Must be
         *                                  at least 1
         * @param stopInterval              The interval to be used to decide whether the induction of rules should be
         *                                  stopped, e.g., a value of 10 means that the rule induction might be stopped
         *                                  after 10, 20, ... rules. Must be a multiple of `updateInterval`
         * @param numPast                   The number of quality scores of past iterations to be stored in a buffer.
         *                                  Must be at least 1
         * @param numCurrent                The number of quality scores of the most recent iterations to be stored in a
         *                                  buffer. Must be at least 1
         * @param minImprovement            The minimum improvement in percent that must be reached for the rule
         *                                  induction to be continued. Must be in [0, 1]
         * @param forceStop                 True, if the induction of rules should be forced to be stopped, if the
         *                                  stopping criterion is met, false, if the time of stopping should only be
         *                                  stored
         */
        MeasureStoppingCriterion(std::unique_ptr<IEvaluationMeasure> measurePtr,
                                 std::unique_ptr<IAggregationFunction> aggregationFunctionPtr, uint32 minRules,
                                 uint32 updateInterval, uint32 stopInterval, uint32 numPast, uint32 numCurrent,
                                 float64 minImprovement, bool forceStop);

        Result test(const IPartition& partition, const IStatistics& statistics, uint32 numRules) override;

};
