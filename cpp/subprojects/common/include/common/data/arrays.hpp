/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include <algorithm>


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
