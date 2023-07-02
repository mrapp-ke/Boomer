/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <algorithm>

namespace boosting {

    /**
     * Adds the elements in an array `b` to the elements in another array `a`, such that `a = a + b`.
     *
     * @tparam T            The type of the arrays `a` and `b`
     * @param a             A pointer to an array of template type `T` to be updated
     * @param b             A pointer to an array of template type `T`
     * @param numElements   The number of elements in the arrays `a` and `b`
     *
     */
    template<typename T>
    static inline void addToArray(T* a, const T* b, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] += b[i];
        }
    }

    /**
     * Adds the elements in an array `b` to the elements in another array `a`. The elements in the array `b` are
     * multiplied by a given weight, such that `a = a + (b * weight)`.
     *
     * @tparam T            The type of the arrays `a` and `b`
     * @tparam W            The type of the weight
     * @param a             A pointer to an array of template type `T` to be updated
     * @param b             A pointer to an array of template type `T`
     * @param numElements   The number of elements in the arrays `a` and `b`
     * @param weight        The weight, the elements in the array `b` should be multiplied by
     *
     */
    template<typename T, typename W>
    static inline void addToArray(T* a, const T* b, uint32 numElements, W weight) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] += (b[i] * weight);
        }
    }

    /**
     * Adds the elements in an array `b` to the elements in another array `a`, such that `a = a + b`. The indices of
     * elements in the array `b` that correspond to the elements in array `a` are given as an additional array.
     *
     * @param a             A pointer to an array of template type `T` to be updated
     * @param b             A pointer to an array of template type `T`
     * @param numElements   The number of elements in the arrays `a` and `b`
     * @param indices       A pointer to an array of type `uint32` that stores the indices of the elements in the array
     *                      `b` that correspond to the elements in array `a`
     *
     */
    template<typename T>
    static inline void addToArray(T* a, const T* b, const uint32* indices, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] += b[index];
        }
    }

    /**
     * Adds the elements in an array `b` to the elements in another array `a`. The elements in the array `b` are
     * multiplied by a given weight, such that `a = a + (b * weight)`. The indices of elements in the array `b` that
     * correspond to the elements in array `a` are given as an additional array.
     *
     * @tparam T            The type of the arrays `a` and `b`
     * @tparam W            The type of the weight
     * @param a             A pointer to an array of template type `T` to be updated
     * @param b             A pointer to an array of template type `T`
     * @param numElements   The number of elements in the arrays `a` and `b`
     * @param weight        The weight, the elements in the array `b` should be multiplied by
     * @param indices       A pointer to an array of type `uint32` that stores the indices of the elements in the array
     *                      `b` that correspond to the elements in array `a`
     *
     */
    template<typename T, typename W>
    static inline void addToArray(T* a, const T* b, const uint32* indices, uint32 numElements, W weight) {
        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indices[i];
            a[i] += (b[index] * weight);
        }
    }

    /**
     * Removes the elements in an array `b` from the elements in another array `a`, such that `a = a - b`.
     *
     * @tparam T            The type of the arrays `a` and `b`
     * @param a             A pointer to an array of template type `T` to be updated
     * @param b             A pointer to an array of template type `T`
     * @param numElements   The number of elements in the arrays `a` and `b`
     *
     */
    template<typename T>
    static inline void removeFromArray(T* a, const T* b, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] -= b[i];
        }
    }

    /**
     * Removes the elements in an array `b` from the elements in another array `a`. The elements in the array `b` are
     * multiplied by a given weight, such that `a = a - (b * weight)`.
     *
     * @tparam T            The type of the arrays `a` and `b`
     * @tparam W            The type of the weight
     * @param a             A pointer to an array of template type `T` to be updated
     * @param b             A pointer to an array of template type `T`
     * @param numElements   The number of elements in the arrays `a` and `b`
     * @param weight        The weight, the elements in the array `b` should be multiplied by
     *
     */
    template<typename T, typename W>
    static inline void removeFromArray(T* a, const T* b, uint32 numElements, W weight) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] -= (b[i] * weight);
        }
    }

}
