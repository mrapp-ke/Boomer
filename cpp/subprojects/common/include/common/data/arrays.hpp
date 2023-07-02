/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"

#include <algorithm>
#include <cstddef>

/**
 * Sets all elements in an array to zero.
 *
 * @tparam T            The type of the array
 * @param a             A pointer to an array of template type `T`
 * @param numElements   The number of elements in the array
 */
template<typename T>
static inline void setArrayToZeros(T* a, uint32 numElements) {
    std::fill(a, a + numElements, 0);
}

/**
 * Sets all elements in an array to a specific value.
 *
 * @tparam T            The type of the array
 * @param a             A pointer to an array of template type `T`
 * @param numElements   The number of elements in the array
 * @param value         The value to be set
 */
template<typename T>
static inline void setArrayToValue(T* a, uint32 numElements, T value) {
    std::fill(a, a + numElements, value);
}

/**
 * Sets the elements in an array to increasing values.
 *
 * @tparam T            The type of the array
 * @param a             A pointer to an array of template type `T`
 * @param numElements   The number of elements in the array
 * @param start         The value to start at
 * @param increment     The difference between the values
 */
template<typename T>
static inline void setArrayToIncreasingValues(T* a, uint32 numElements, T start, T increment) {
    T nextValue = start;

    for (uint32 i = 0; i < numElements; i++) {
        a[i] = nextValue;
        nextValue += increment;
    }
}

/**
 * Copy all elements from one array another one.
 *
 * @tparam T            The type of the arrays
 * @param from          A pointer to an array of template type `T` to be copied
 * @param to            A pointer to an array of template type `T`, the elements should be copied to
 * @param numElements   The number of elements to be copied
 */
template<typename T>
static inline void copyArray(const T* from, T* to, uint32 numElements) {
    for (uint32 i = 0; i < numElements; i++) {
        to[i] = from[i];
    }
}

/**
 * Copy all elements from an iterator to an array.
 *
 * @tparam FromIterator The type of the iterator to copy from
 * @tparam T            The type of the array to copy to
 * @param from          The iterator to copy from
 * @param to            The array to copy to
 * @param numElements   The number of elements to be copied
 */
template<typename FromIterator, typename T>
static inline void copyArray(FromIterator from, T* to, uint32 numElements) {
    for (uint32 i = 0; i < numElements; i++) {
        to[i] = from[i];
    }
}

/**
 * Sets all elements in an array `a` to the difference between the elements in two other arrays `b` and `c`, such that
 * `a = b - c`.
 *
 * @tparam T            The type of the arrays `a`, `b` and `c`
 * @param a             A pointer to an array of template type `T` to be updated
 * @param b             A pointer to an array of template type `T`
 * @param c             A pointer to an array of template type `T`
 * @param numElements   The number of elements in the arrays `a`, `b` and `c`
 */
template<typename T>
static inline void setArrayToDifference(T* a, const T* b, const T* c, uint32 numElements) {
    for (uint32 i = 0; i < numElements; i++) {
        a[i] = b[i] - c[i];
    }
}

/**
 * Sets all elements in an array `a` to the difference between the elements in two other array `b` and `c`, such that
 * `a = b - c`. The indices of elements in the array `b` that correspond to the elements in arrays `a` and `c` are given
 * as an additional array.
 *
 * @tparam T            The type of the arrays `a`, `b` and `c`
 * @param a             A pointer to an array of template type `T` to be updated
 * @param b             A pointer to an array of template type `T`
 * @param c             A pointer to an array of template type `T`
 * @param indices       A pointer to an array of type `uint32` that stores the indices of the elements in the array `b`
 *                      that correspond to the elements in arrays `a` and `c`
 * @param numElements   The number of elements in the array `a`
 */
template<typename T>
static inline void setArrayToDifference(T* a, const T* b, const T* c, const uint32* indices, uint32 numElements) {
    for (uint32 i = 0; i < numElements; i++) {
        uint32 index = indices[i];
        a[i] = b[index] - c[i];
    }
}

/**
 * Calculates and returns a hash value from an array of type `uint32`.
 *
 * @param a             A pointer to an array of type `uint32`
 * @param numElements   The number of elements in the array
 * @return              The hash value
 */
static inline constexpr std::size_t hashArray(const uint32* a, uint32 numElements) {
    std::size_t hashValue = (std::size_t) numElements;

    for (uint32 i = 0; i < numElements; i++) {
        hashValue ^= a[i] + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
    }

    return hashValue;
}

/**
 * Returns whether two arrays are equal or not.
 *
 * @tparam T        The type of the arrays
 * @param first     A pointer to an array of template type `T`
 * @param numFirst  The number of elements in the array `first`
 * @param second    A pointer to another array of template type `T`
 * @param numSecond The number of elements in the array `second`
 * @return          True, if both arrays are equal, false otherwise
 */
template<typename T>
static inline constexpr bool compareArrays(const T* first, uint32 numFirst, const T* second, uint32 numSecond) {
    if (numFirst != numSecond) {
        return false;
    }

    for (uint32 i = 0; i < numFirst; i++) {
        if (first[i] != second[i]) {
            return false;
        }
    }

    return true;
}
