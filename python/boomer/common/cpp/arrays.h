/**
 * Provides type definitions consistent to those used in `arrays.pxd`, as well as utility functions for allocating
 * arrays.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include <cstdint>

typedef uint8_t uint8;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef intptr_t intp;
typedef float float32;
typedef double float64;


namespace arrays {

    /**
     * Sets all elements in an one- or two-dimensional array to zero.
     *
     * @param a             A pointer to an array of template type `T`
     * @param numElements   The number of elements in the array
     */
    template<typename T>
    static inline void setToZeros(T* a, uint32 numElements) {
        for (uint32 i = 0; i < numElements; i++) {
            a[i] = 0;
        }
    }

}
