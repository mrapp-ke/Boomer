/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <cmath>
#include <limits>

typedef long int int64;
typedef unsigned char uint8;
typedef unsigned int uint32;
typedef float float32;
typedef double float64;

/**
 * Returns whether two floating point values `a` and `b` are (approximately) equal.
 *
 * @tparam T    The type of the floating point values to be compared
 * @param a     The first floating point value
 * @param b     The second floating point value
 * @return      True if the given floating point values are equal, false otherwise
 */
template<typename T>
static inline constexpr bool isEqual(T a, T b) {
    return std::fabs(a - b) <= std::numeric_limits<T>::epsilon() * std::fmax(1, std::fmax(std::fabs(a), std::fabs(b)));
}
