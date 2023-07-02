/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/ring_buffer.hpp"
#include "common/math/math.hpp"

#include <memory>

/**
 * Defines an interface for all classes that allow to aggregate the values that are stored in a buffer.
 */
class IAggregationFunction {
    public:

        virtual ~IAggregationFunction() {};

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
 * An implementation of the type `IAggregationFunction` that aggregates the values that are stored in a buffer by
 * finding the minimum value.
 */
class MinAggregationFunction final : public IAggregationFunction {
    public:

        float64 aggregate(RingBuffer<float64>::const_iterator begin,
                          RingBuffer<float64>::const_iterator end) const override {
            uint32 numElements = end - begin;
            float64 min = begin[0];

            for (uint32 i = 1; i < numElements; i++) {
                float64 value = begin[i];

                if (value < min) {
                    min = value;
                }
            }

            return min;
        }
};

/**
 * An implementation of the type `IAggregationFunction` that aggregates the values that are stored in a buffer by
 * finding the maximum value.
 */
class MaxAggregationFunction final : public IAggregationFunction {
    public:

        float64 aggregate(RingBuffer<float64>::const_iterator begin,
                          RingBuffer<float64>::const_iterator end) const override {
            uint32 numElements = end - begin;
            float64 max = begin[0];

            for (uint32 i = 1; i < numElements; i++) {
                float64 value = begin[i];

                if (value > max) {
                    max = value;
                }
            }

            return max;
        }
};

/**
 * An implementation of the type `IAggregationFunction` that aggregates the values that are stored in a buffer by
 * calculating the arithmetic mean.
 */
class ArithmeticMeanAggregationFunction final : public IAggregationFunction {
    public:

        float64 aggregate(RingBuffer<float64>::const_iterator begin,
                          RingBuffer<float64>::const_iterator end) const override {
            uint32 numElements = end - begin;
            float64 mean = 0;

            for (uint32 i = 0; i < numElements; i++) {
                float64 value = begin[i];
                mean = iterativeArithmeticMean<float64>(i + 1, value, mean);
            }

            return mean;
        }
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IAggregationFunction`.
 */
class IAggregationFunctionFactory {
    public:

        virtual ~IAggregationFunctionFactory() {};

        /**
         * Creates and returns a new object of type `IAggregationFunction`.
         *
         * @return An unique pointer to an object of type `IAggregationFunction` that has been created
         */
        virtual std::unique_ptr<IAggregationFunction> create() const = 0;
};

/**
 * Allows to create instances of the type `IAggregationFunction` that aggregate the values that are stored in a buffer
 * by finding the minimum value.
 */
class MinAggregationFunctionFactory final : public IAggregationFunctionFactory {
    public:

        std::unique_ptr<IAggregationFunction> create() const override {
            return std::make_unique<MinAggregationFunction>();
        }
};

/**
 * Allows to create instances of the type `IAggregationFunction` that aggregate the values that are stored in a buffer
 * by finding the maximum value.
 */
class MaxAggregationFunctionFactory final : public IAggregationFunctionFactory {
    public:

        std::unique_ptr<IAggregationFunction> create() const override {
            return std::make_unique<MaxAggregationFunction>();
        }
};

/**
 * Allows to create instances of the type `IAggregationFunction` that aggregate the values that are stored in a buffer
 * by calculating the arithmetic mean.
 */
class ArithmeticMeanAggregationFunctionFactory final : public IAggregationFunctionFactory {
    public:

        std::unique_ptr<IAggregationFunction> create() const override {
            return std::make_unique<ArithmeticMeanAggregationFunction>();
        }
};

/**
 * Creates and returns a new object of type `IAggregationFunctionFactory` according to a given `AggregationFunction`.
 *
 * @param aggregationFunction   A value of the enum `AggregationFunction`
 * @return                      An unique pointer to an object of type `IAggregationFunctionFactory` that has been
 *                              created
 */
std::unique_ptr<IAggregationFunctionFactory> createAggregationFunctionFactory(AggregationFunction aggregationFunction) {
    switch (aggregationFunction) {
        case AggregationFunction::MIN:
            return std::make_unique<MinAggregationFunctionFactory>();
        case AggregationFunction::MAX:
            return std::make_unique<MaxAggregationFunctionFactory>();
        default:
            return std::make_unique<ArithmeticMeanAggregationFunctionFactory>();
    }
}
