/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/iterator/diagonal_iterator.hpp"

namespace boosting {

    /**
     * Implements row-wise read-only access to the gradients and Hessians that have been calculated using a
     * non-decomposable loss function and are stored in pre-allocated C-contiguous arrays.
     */
    class DenseExampleWiseStatisticConstView {
        protected:

            /**
             * The number of rows in the view.
             */
            const uint32 numRows_;

            /**
             * The number of gradients per row.
             */
            const uint32 numGradients_;

            /**
             * The number of Hessians per row.
             */
            const uint32 numHessians_;

            /**
             * A pointer to an array that stores the gradients.
             */
            float64* gradients_;

            /**
             * A pointer to an array that stores the Hessians.
             */
            float64* hessians_;

        public:

            /**
             * @param numRows       The number of rows in the view
             * @param numGradients  The number of gradients per row
             * @param numHessians   The number of Hessians per row
             * @param gradients     A pointer to an array of type `float64` that stores the gradients, the view provides
             *                      access to
             * @param hessians      A pointer to an array of type `float64` that stores the Hessians, the view provides
             *                      access to
             */
            DenseExampleWiseStatisticConstView(uint32 numRows, uint32 numGradients, uint32 numHessians,
                                               float64* gradients, float64* hessians);

            virtual ~DenseExampleWiseStatisticConstView() {};

            /**
             * An iterator that provides read-only access to the gradients.
             */
            typedef const float64* gradient_const_iterator;

            /**
             * An iterator that provides read-only access to the Hessians.
             */
            typedef const float64* hessian_const_iterator;

            /**
             * An iterator that provides read-only access to the Hessians that correspond to the diagonal of the matrix.
             */
            typedef DiagonalConstIterator<float64> hessian_diagonal_const_iterator;

            /**
             * Returns a `gradient_const_iterator` to the beginning of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_const_iterator` to the beginning of the given row
             */
            gradient_const_iterator gradients_cbegin(uint32 row) const;

            /**
             * Returns a `gradient_const_iterator` to the end of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_const_iterator` to the end of the given row
             */
            gradient_const_iterator gradients_cend(uint32 row) const;

            /**
             * Returns a `hessian_const_iterator` to the beginning of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_const_iterator` to the beginning of the given row
             */
            hessian_const_iterator hessians_cbegin(uint32 row) const;

            /**
             * Returns a `hessian_const_iterator` to the end of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_const_iterator` to the end of the given row
             */
            hessian_const_iterator hessians_cend(uint32 row) const;

            /**
             * Returns a `hessian_diagonal_const_iterator` to the beginning of the Hessians that correspond to the
             * diagonal of the Hessian matrix at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_diagonal_const_iterator` to the beginning
             */
            hessian_diagonal_const_iterator hessians_diagonal_cbegin(uint32 row) const;

            /**
             * Returns a `hessian_diagonal_const_iterator` to the end of the Hessians that correspond to the diagonal of
             * the Hessian matrix at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_diagonal_const_iterator` to the end
             */
            hessian_diagonal_const_iterator hessians_diagonal_cend(uint32 row) const;

            /**
             * Returns the number of rows in the view.
             *
             * @return The number of rows
             */
            uint32 getNumRows() const;

            /**
             * Returns the number of columns in the view.
             *
             * @return The number of columns
             */
            uint32 getNumCols() const;
    };

    /**
     * Implements row-wise read and write access to the gradients and Hessians that have been calculated using a
     * non-decomposable loss function and are stored in pre-allocated C-contiguous arrays.
     */
    class DenseExampleWiseStatisticView : public DenseExampleWiseStatisticConstView {
        public:

            /**
             * @param numRows       The number of rows in the view
             * @param numGradients  The number of gradients per row
             * @param numHessians   The number of Hessians per row
             * @param gradients     A pointer to an array of type `float64` that stores the gradients, the view provides
             *                      access to
             * @param hessians      A pointer to an array of type `float64` that stores the Hessians, the view provides
             *                      access to
             */
            DenseExampleWiseStatisticView(uint32 numRows, uint32 numGradients, uint32 numHessians, float64* gradients,
                                          float64* hessians);

            virtual ~DenseExampleWiseStatisticView() override {};

            /**
             * An iterator that provides access to the gradients and allows to modify them.
             */
            typedef float64* gradient_iterator;

            /**
             * An iterator that provides access to the Hessians and allows to modify them.
             */
            typedef float64* hessian_iterator;

            /**
             * Returns a `gradient_iterator` to the beginning of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_iterator` to the beginning of the given row
             */
            gradient_iterator gradients_begin(uint32 row);

            /**
             * Returns a `gradient_iterator` to the end of the gradients at a specific row.
             *
             * @param row   The row
             * @return      A `gradient_iterator` to the end of the given row
             */
            gradient_iterator gradients_end(uint32 row);

            /**
             * Returns a `hessian_iterator` to the beginning of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_iterator` to the beginning of the given row
             */
            hessian_iterator hessians_begin(uint32 row);

            /**
             * Returns a `hessian_iterator` to the end of the Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `hessian_iterator` to the end of the given row
             */
            hessian_iterator hessians_end(uint32 row);

            /**
             * Sets all gradients and Hessians in the matrix to zero.
             */
            void clear();

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
