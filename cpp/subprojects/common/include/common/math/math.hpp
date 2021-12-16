/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include <cmath>


/**
 * Returns the result of the floating point division `numerator / denominator` or 0, if a division by zero occurs.
 *
 * @tparam T            The type of the operands
 * @param numerator     The numerator
 * @param denominator   The denominator
 * @return              The result of the division or 0, if a division by zero occurred
 */
template<typename T>
static inline constexpr T divideOrZero(T numerator, T denominator) {
    T result = numerator / denominator;
    return std::isfinite(result) ? result : 0;
}

/**
 * Calculates the arithmetic mean of two values `small` and `large`, where `small < large`.
 *
 * The mean is calculated as `small + ((large - small) * 0.5`, instead of `(small + large) / 2`, to prevent overflows.
 *
 * @param small The smaller of both values
 * @param large The larger of both values
 * @return      The mean that has been calculated
 */
template<typename T>
static inline constexpr T arithmeticMean(T small, T large) {
    return small + ((large - small) * 0.5);
}

/**
 * Allows to compute the arithmetic mean of several floating point values `x_1, ..., x_n` in an iterative manner, which
 * prevents overflows.
 *
 * This function must be invoked for each value as follows:
 * `mean_1 = iterativeArithmeticMean(1, x_1, 0); ...; mean_n = iterativeArithmeticMean(n, x_n, mean_n-1)`
 *
 * @tparam T    The type of the values
 * @param n     The index of the value, starting at 1
 * @param x     The n-th value
 * @param mean  The arithmetic mean of all previously provided values
 * @return      The arithmetic mean of all values provided so far
 */
template<typename T>
static inline constexpr T iterativeArithmeticMean(uint32 n, T x, T mean) {
    return mean + ((x - mean) / (T) n);
}
