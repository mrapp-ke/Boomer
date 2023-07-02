/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include "common/macros.hpp"

/**
 * Specifies different types of aggregation functions that allow to aggregate the values that are stored in a buffer.
 */
enum MLRLCOMMON_API AggregationFunction : uint8 {
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
