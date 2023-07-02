/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/tuple.hpp"

namespace boosting {

    /**
     * Implements row-wise read-only access to the gradients and Hessians that have been calculated using a label-wise
     * decomposable loss function and are stored in pre-allocated C-contiguous arrays.
     */
    class DenseLabelWiseStatisticConstView {
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
             * A pointer to an array that stores the gradients and Hessians.
             */
            Tuple<float64>* statistics_;

        public:

            /**
             * @param numRows       The number of rows in the view
             * @param numCols       The number of columns in the view
             * @param statistics    A pointer to a C-contiguous array fo type `Tuple<float64>` that stores the gradients
             *                      and Hessians, the view provides access to
             */
            DenseLabelWiseStatisticConstView(uint32 numRows, uint32 numCols, Tuple<float64>* statistics);

            virtual ~DenseLabelWiseStatisticConstView() {};

            /**
             * An iterator that provides read-only access to the elements in the view.
             */
            typedef const Tuple<float64>* const_iterator;

            /**
             * Returns a `const_iterator` to the beginning of a specific row.
             *
             * @param row   The row
             * @return      A `const_iterator` to the beginning
             */
            const_iterator cbegin(uint32 row) const;

            /**
             * Returns a `const_iterator` to the end of a specific row.
             *
             * @param row   The row
             * @return      A `const_iterator` to the end
             */
            const_iterator cend(uint32 row) const;

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
     * label-wise decomposable loss function and are stored in pre-allocated C-contiguous arrays.
     */
    class DenseLabelWiseStatisticView : public DenseLabelWiseStatisticConstView {
        public:

            /**
             * @param numRows       The number of rows in the view
             * @param numCols       The number of columns in the view
             * @param statistics    A pointer to a C-contiguous array fo type `Tuple<float64>` that stores the gradients
             *                      and Hessians, the view provides access to
             */
            DenseLabelWiseStatisticView(uint32 numRows, uint32 numCols, Tuple<float64>* statistics);

            virtual ~DenseLabelWiseStatisticView() override {};

            /**
             * An iterator that provides access to the elements in the view and allows to modify them.
             */
            typedef Tuple<float64>* iterator;

            /**
             * Returns an `iterator` to the beginning of a specific row.
             *
             * @param row   The row
             * @return      An `iterator` to the beginning
             */
            iterator begin(uint32 row);

            /**
             * Returns an `iterator` to the end of a specific row.
             *
             * @param row   The row
             * @return      An `iterator` to the end
             */
            iterator end(uint32 row);

            /**
             * Sets all gradients and Hessians in the matrix to zero.
             */
            void clear();

            /**
             * Adds all gradients and Hessians in a vector to a specific row of this matrix. The gradients and Hessians
             * to be added are multiplied by a specific weight.
             *
             * @param row       The row
             * @param begin     A `const_iterator` to the beginning of the vector
             * @param end       A `const_iterator` to the end of the vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToRow(uint32 row, const_iterator begin, const_iterator end, float64 weight);
    };

}
