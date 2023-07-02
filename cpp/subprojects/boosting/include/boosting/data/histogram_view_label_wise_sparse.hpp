/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/data/statistic_view_label_wise_sparse.hpp"
#include "common/data/triple.hpp"

namespace boosting {

    /**
     * Implements row-wise read-only access to the gradients and Hessians that have been calculated using a label-wise
     * decomposable loss function and are stored in a pre-allocated histogram in the list of lists (LIL) format.
     */
    class SparseLabelWiseHistogramConstView {
        protected:

            /**
             * The number of rows in the view.
             */
            const uint32 numRows_;

            /**
             * The number of columns in the view.
             */
            const uint32 numCols_;

            /**
             * A pointer to an array that stores the gradients and Hessians of each bin.
             */
            Triple<float64>* statistics_;

            /**
             * A pointer to an array that stores the weight of each bin.
             */
            float64* weights_;

        public:

            /**
             * @param numRows       The number of rows in the view
             * @param numCols       The number of columns in the view
             * @param statistics    A pointer to an array that stores the gradients and Hessians of each bin
             * @param weights       A pointer to an array that stores the weight of each bin
             */
            SparseLabelWiseHistogramConstView(uint32 numRows, uint32 numCols, Triple<float64>* statistics,
                                              float64* weights);

            virtual ~SparseLabelWiseHistogramConstView() {};

            /**
             * An iterator that provides read-only access to the gradients and Hessians.
             */
            typedef const Triple<float64>* const_iterator;

            /**
             * An iterator that provides read-only access to the weights that correspond to individual bins.
             */
            typedef const float64* weight_const_iterator;

            /**
             * Returns a `const_iterator` to the beginning of the gradients and Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `const_iterator` to the beginning of the row
             */
            const_iterator cbegin(uint32 row) const;

            /**
             * Returns a `const_iterator` to the end of the gradients and Hessians at a specific row.
             *
             * @param row   The row
             * @return      A `const_iterator` to the end of the row
             */
            const_iterator cend(uint32 row) const;

            /**
             * Returns a `weight_const_iterator` to the beginning of the weights that correspond to individual bins.
             *
             * @return A `weight_const_iterator` to the beginning
             */
            weight_const_iterator weights_cbegin() const;

            /**
             * Returns a `weight_const_iterator` to the end of the weights that correspond to individual bins.
             *
             * @return A `weight_const_iterator` to the end
             */
            weight_const_iterator weights_cend() const;

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
     * label-wise decomposable loss function and are stored in a pre-allocated histogram in the list of lists (LIL)
     * format.
     */
    class SparseLabelWiseHistogramView : public SparseLabelWiseHistogramConstView {
        public:

            /**
             * @param numRows       The number of rows in the view
             * @param numCols       The number of columns in the view
             * @param statistics    A pointer to an array that stores the gradients and Hessians of each bin
             * @param weights       A pointer to an array that stores the weight of each bin
             */
            SparseLabelWiseHistogramView(uint32 numRows, uint32 numCols, Triple<float64>* statistics, float64* weights);

            virtual ~SparseLabelWiseHistogramView() override {};

            /**
             * Sets all gradients and Hessians in the matrix to zero.
             */
            void clear();

            /**
             * Adds all gradients and Hessians in a vector to a specific row of this histogram. The gradients and
             * Hessians to be added are multiplied by a specific weight.
             *
             * @param row       The row
             * @param begin     A `SparseLabelWiseStatisticConstView::const_iterator` to the beginning of the vector
             * @param end       A `SparseLabelWiseStatisticConstView::const_iterator` to the end of the vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToRow(uint32 row, SparseLabelWiseStatisticConstView::const_iterator begin,
                          SparseLabelWiseStatisticConstView::const_iterator end, float64 weight);
    };

}
