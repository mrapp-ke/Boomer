/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"

namespace boosting {

    /**
     * Allows to execute LAPACK routines.
     */
    class Lapack final {
        public:

            /**
             * A function pointer to LAPACK'S DSYSV routine.
             */
            typedef void (*DsysvFunction)(char* uplo, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b,
                                          int* ldb, double* work, int* lwork, int* info);

        private:

            const DsysvFunction dsysvFunction_;

        public:

            /**
             * @param dsysvFunction A function pointer to LAPACK's DSYSV routine
             */
            Lapack(DsysvFunction dsysvFunction);

            /**
             * Determines and returns the optimal value for the parameter "lwork" as used by LAPACK'S DSYSV routine.
             *
             * This function must be run before attempting to solve a linear system using the function `dsysv` to
             * determine the optimal value for the parameter "lwork".
             *
             * @param tmpArray1 A pointer to an array of type `float64`, shape `(n, n)` that will be used by the
             *                  function `dsysv` to temporarily store values computed by the DSYSV routine. May contain
             *                  arbitrary values
             * @param output    A pointer to an array of type `float64`, shape `(n)`, the solution of the system of
             *                  linear equations should be written to by the function `dsysv`. May contain arbitrary
             *                  values
             * @param n         The number of equations in the linear system to be solved by the function `dsysv`
             * @return          The optimal value for the parameter "lwork"
             */
            int queryDsysvLworkParameter(float64* tmpArray1, float64* output, int n) const;

            /**
             * Computes and returns the solution to a linear system A * X = B using LAPACK's DSYSV solver (see
             * http://www.netlib.org/lapack/explore-html/d6/d0e/group__double_s_ysolve_ga9995c47692c9885ed5d6a6b431686f41.html).
             *
             * The function `queryDsysvLworkParameter` must be run beforehand to determine the optimal value for the
             * parameter "lwork" and to allocate a temporary array depending on this value.
             *
             * DSYSV requires A to be a matrix with shape `(n, n)`, representing the coefficients, and B to be a matrix
             * with shape `(n, nrhs)`, representing the ordinates. X is a matrix of unknowns with shape `(n, nrhs)`.
             *
             * DSYSV will overwrite the matrices A and B. When terminated successfully, B will contain the solution to
             * the system of linear equations. To retain their state, this function will copy the given arrays before
             * invoking DSYSV.
             *
             * Furthermore, DSYSV assumes the matrix of coefficients A to be symmetrical, i.e., it will only use the
             * upper-right triangle of A, whereas the remaining elements are ignored. For reasons of space efficiency,
             * this function expects the coefficients to be given as an array with shape `n * (n + 1) / 2`, representing
             * the elements of the upper-right triangle of A, where the columns are appended to each other and
             * unspecified elements are omitted. This function will implicitly convert the given array into a matrix
             * that is suited for DSYSV.
             *
             * @param tmpArray1                 A pointer to an array of type `float64`, shape `(n, n)` that stores the
             *                                  coefficients in the matrix A. It will be used to temporarily store
             *                                  values computed by the DSYSV routine
             * @param tmpArray2                 A pointer to an array of type `int`, shape `(n)` that will be used to
             *                                  temporarily store values computed by the DSYSV routine. May contain
             *                                  arbitrary values
             * @param tmpArray3                 A pointer to an array of type `double`, shape `(lwork)` that will be
             *                                  used to temporarily store values computed by the DSYSV routine. May
             *                                  contain arbitrary values
             * @param output                    A pointer to an array of type `float64`, shape `(n)` that stores the
             *                                  ordinates in the matrix A. The solution of the system of linear
             *                                  equations will be written to this array
             * @param n                         The number of equations
             * @param lwork                     The value for the parameter "lwork" to be used by the DSYSV routine.
             *                                  Must have been determined using the function `queryDsysvLworkParameter`
             */
            void dsysv(float64* tmpArray1, int* tmpArray2, double* tmpArray3, float64* output, int n, int lwork) const;
    };

}
