/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/math/math.hpp"

namespace boosting {

    /**
     * Copies Hessians from an iterator to a matrix of coefficients that may be passed to LAPACK's DSYSV routine. Only
     * the Hessians that correspond to the indices in a second iterator are taken into account.
     *
     * @tparam HessianIterator  The type of the iterator that provides access to the Hessians
     * @tparam IndexIterator    The type of the iterator that provides access to the indices
     * @param hessianIterator   An iterator that provides random access to the Hessians
     * @param indexIterator     An iterator that provides random access to the indices
     * @param coefficients      A pointer to an array of type `float64`, shape `(n * n)`, the Hessians should be copied
     *                          to
     * @param n                 The dimensionality of the matrix of coefficients
     */
    template<typename HessianIterator, typename IndexIterator>
    static inline void copyCoefficients(HessianIterator hessianIterator, IndexIterator indexIterator,
                                        float64* coefficients, uint32 n) {
        for (uint32 c = 0; c < n; c++) {
            uint32 offset = c * n;
            uint32 offset2 = triangularNumber(indexIterator[c]);

            for (uint32 r = 0; r <= c; r++) {
                coefficients[offset + r] = hessianIterator[offset2 + indexIterator[r]];
            }
        }
    }

}
