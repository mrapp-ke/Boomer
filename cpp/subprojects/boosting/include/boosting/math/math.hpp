/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <cmath>


namespace boosting {

    /**
     * Returns the result of the floating point division `numerator / denominator` or 0, if a division by zero occurs.
     *
     * @tparam T            The type of the operands
     * @param numerator     The numerator
     * @param denominator   The denominator
     * @return              The result of the division or 0, if a division by zero occurred
     */
    template<class T>
    static inline T divideOrZero(T numerator, T denominator) {
        T result = numerator / denominator;
        return std::isfinite(result) ? result : 0;
    }

    /**
     * Calculates and returns the n-th triangular number, i.e., the number of elements in a n times n triangle.
     *
     * @param n A scalar of type `uint32`, representing the order of the triangular number
     * @return  A scalar of type `uint32`, representing the n-th triangular number
     */
    static inline uint32 triangularNumber(uint32 n) {
        return (n * (n + 1)) / 2;
    }

    /**
     * Calculates and returns the logistic function `1 / (1 + exp(-x))`, given a specific value `x`.
     *
     * This implementation exploits the identity `1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))` to increase numerical
     * stability (see, e.g., section "Numerically stable sigmoid function" in
     * https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/).
     *
     * @param x The value `x`
     * @return  The value that has been calculated
     */
    static inline float64 logisticFunction(float64 x) {
        if (x >= 0) {
            float64 exponential = std::exp(-x);  // Evaluates to 0 for large x, resulting in 1 ultimately
            return 1 / (1 + exponential);
        } else {
            float64 exponential = std::exp(x);  // Evaluates to 0 for large x, resulting in 0 ultimately
            return exponential / (1 + exponential);
        }
    }

    /**
     * Computes and returns the square of the L2 norm of a specific vector, i.e. the sum of the squares of its elements.
     * To obtain the actual L2 norm, the square-root of the result provided by this function must be computed.
     *
     * @tparam Iterator The type of the iterator that provides access to the elements in the vector
     * @param iterator  An iterator of template type `Iterator` that provides random access to the elements in the
     *                  vector
     * @param n         The number of elements in the vector
     * @return          The square of the L2 norm
    */
    template<class Iterator>
    static inline float64 l2NormPow(Iterator iterator, uint32 n) {
        float64 result = 0;

        for (uint32 i = 0; i < n; i++) {
            float64 value = iterator[i];
            result += (value * value);
        }

        return result;
    }

}
