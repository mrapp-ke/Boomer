/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


namespace boosting {

    /**
     * A two-dimensional matrix that stores gradients and Hessians that have been calculated using a non-decomposable
     * loss function in C-contiguous arrays. For each element at a certain row a single gradient, but multiple Hessians
     * are stored. In a vector that stores `n` gradients `(n * (n + 1)) / 2` Hessians are stored. The Hessians can be
     * viewed as a symmetric Hessian matrix with `n` rows and columns.
     */
    class DenseExampleWiseStatisticMatrix final {

        private:

            uint32 numRows_;

            uint32 numGradients_;

            uint32 numHessians_;

            float64* gradients_;

            float64* hessians_;

        public:

            /**
             * @param numRows       The number of rows in the matrix
             * @param numGradients  The number of gradients per row
             */
            DenseExampleWiseStatisticMatrix(uint32 numRows, uint32 numGradients);

            /**
             * @param numRows       The number of rows in the matrix
             * @param numGradients  The number of gradients per row
             * @param init          True, if all gradients and Hessians in the matrix should be initialized with zero,
             *                      false otherwise
             */
            DenseExampleWiseStatisticMatrix(uint32 numRows, uint32 numGradients, bool init);

            ~DenseExampleWiseStatisticMatrix();

            /**
             * An iterator that provides access to the gradients in the matrix and allows to modify them.
             */
            typedef float64* gradient_iterator;

            /**
             * An iterator that provides read-only access to the gradients in the matrix.
             */
            typedef const float64* gradient_const_iterator;

            /**
             * An iterator that provides access to the Hessians in the matrix and allows to modify them.
             */
            typedef float64* hessian_iterator;

            /**
             * An iterator that provides read-only access to the Hessians in the matrix.
             */
            typedef const float64* hessian_const_iterator;

            /**
             * Returns a `gradient_iterator` to the beginning of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_iterator` to the beginning of the given row
             */
            gradient_iterator gradients_row_begin(uint32 row);

            /**
             * Returns a `gradient_iterator` to the end of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_iterator` to the end of the given row
             */
            gradient_iterator gradients_row_end(uint32 row);

            /**
             * Returns a `gradient_const_iterator` to the beginning of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_const_iterator` to the beginning of the given row
             */
            gradient_const_iterator gradients_row_cbegin(uint32 row) const;

            /**
             * Returns a `gradient_const_iterator` to the end of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_const_iterator` to the end of the given row
             */
            gradient_const_iterator gradients_row_cend(uint32 row) const;

            /**
             * Returns a `hessian_iterator` to the beginning of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_iterator` to the beginning of the given row
             */
            hessian_iterator hessians_row_begin(uint32 row);

            /**
             * Returns a `hessian_iterator` to the end of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_iterator` to the end of the given row
             */
            hessian_iterator hessians_row_end(uint32 row);

            /**
             * Returns a `hessian_const_iterator` to the beginning of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_const_iterator` to the beginning of the given row
             */
            hessian_const_iterator hessians_row_cbegin(uint32 row) const;

            /**
             * Returns a `hessian_const_iterator` to the end of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_const_iterator` to the end of the given row
             */
            hessian_const_iterator hessians_row_cend(uint32 row) const;

            /**
             * Returns the number of rows in the matrix.
             *
             * @return The number of rows
             */
            uint32 getNumRows() const;

            /**
             * Returns the number of gradients per row.
             *
             * @return The number of gradients
             */
            uint32 getNumCols() const;

            /**
             * Sets all gradients and Hessians in the matrix to zero.
             */
            void setAllToZero();

            /**
             * Adds all gradients and Hessians in a vector to a specific row of this matrix. The gradients and Hessians
             * to be added are multiplied by a specific weight.
             *
             * @param row               The row
             * @param gradientsBegin    A `gradient_const_iterator` to the beginning of the gradients in the vector
             * @param gradientsEnd      A `gradient_const_iterator` to the end of the gradients in the vector
             * @param hessiansBegin     A `hessian_const_iterator` to the beginning of the Hessians in the vector
             * @param hessiansEnd       A `hessian_const_iterator` to the end of the Hessians in the vector
             * @param weight            The weight, the gradients and Hessians should be multiplied by
             */
            void addToRow(uint32 row, gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                          hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd, float64 weight);

    };

}