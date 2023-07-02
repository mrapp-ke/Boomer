/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/data/histogram_view_label_wise_sparse.hpp"
#include "boosting/data/statistic_view_label_wise_sparse.hpp"
#include "common/data/triple.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"

namespace boosting {

    /**
     * An one-dimensional vector that stores aggregated gradients and Hessians that have been calculated using a
     * label-wise decomposable loss function in a C-contiguous array. For each element in the vector a single gradient
     * and Hessian, as well as the sums of the weights of the aggregated gradients and Hessians, is stored.
     */
    class SparseLabelWiseStatisticVector final {
        private:

            /**
             * An iterator that provides random read-only access to the statistics in a
             * `SparseLabelWiseStatisticVector`.
             */
            class ConstIterator final {
                private:

                    const Triple<float64>* iterator_;

                    const float64 sumOfWeights_;

                public:

                    /**
                     * @param iterator      An iterator that provides access to the elements in a
                     *                      `SparseLabelWiseStatisticVector`
                     * @param sumOfWeights  The sum of the weights of all statistics that have been added to the vector
                     */
                    ConstIterator(const Triple<float64>* iterator, float64 sumOfWeights);

                    /**
                     * The type that is used to represent the difference between two iterators.
                     */
                    typedef int difference_type;

                    /**
                     * The type of the elements, the iterator provides access to.
                     */
                    typedef const Tuple<float64> value_type;

                    /**
                     * The type of a pointer to an element, the iterator provides access to.
                     */
                    typedef const Tuple<float64>* pointer;

                    /**
                     * The type of a reference to an element, the iterator provides access to.
                     */
                    typedef const Tuple<float64>& reference;

                    /**
                     * The tag that specifies the capabilities of the iterator.
                     */
                    typedef std::random_access_iterator_tag iterator_category;

                    /**
                     * Returns the element at a specific index.
                     *
                     * @param index The index of the element to be returned
                     * @return      The element at the given index
                     */
                    value_type operator[](uint32 index) const;

                    /**
                     * Returns the element, the iterator currently refers to.
                     *
                     * @return The element, the iterator currently refers to
                     */
                    value_type operator*() const;

                    /**
                     * Returns an iterator to the next element.
                     *
                     * @return A reference to an iterator that refers to the next element
                     */
                    ConstIterator& operator++();

                    /**
                     * Returns an iterator to the next element.
                     *
                     * @return A reference to an iterator that refers to the next element
                     */
                    ConstIterator& operator++(int n);

                    /**
                     * Returns an iterator to the previous element.
                     *
                     * @return A reference to an iterator that refers to the previous element
                     */
                    ConstIterator& operator--();

                    /**
                     * Returns an iterator to the previous element.
                     *
                     * @return A reference to an iterator that refers to the previous element
                     */
                    ConstIterator& operator--(int n);

                    /**
                     * Returns whether this iterator and another one refer to the same element.
                     *
                     * @param rhs   A reference to another iterator
                     * @return      True, if the iterators do not refer to the same element, false otherwise
                     */
                    bool operator!=(const ConstIterator& rhs) const;

                    /**
                     * Returns whether this iterator and another one refer to the same element.
                     *
                     * @param rhs   A reference to another iterator
                     * @return      True, if the iterators refer to the same element, false otherwise
                     */
                    bool operator==(const ConstIterator& rhs) const;

                    /**
                     * Returns the difference between this iterator and another one.
                     *
                     * @param rhs   A reference to another iterator
                     * @return      The difference between the iterators
                     */
                    difference_type operator-(const ConstIterator& rhs) const;
            };

            const uint32 numElements_;

            Triple<float64>* statistics_;

            float64 sumOfWeights_;

        public:

            /**
             * @param numElements The number of gradients and Hessians in the vector
             */
            SparseLabelWiseStatisticVector(uint32 numElements);

            /**
             * @param numElements   The number of gradients and Hessians in the vector
             * @param init          True, if all gradients and Hessians in the vector should be initialized with zero,
             *                      false otherwise
             */
            SparseLabelWiseStatisticVector(uint32 numElements, bool init);

            /**
             * @param vector A reference to an object of type `SparseLabelWiseStatisticVector` to be copied
             */
            SparseLabelWiseStatisticVector(const SparseLabelWiseStatisticVector& vector);

            ~SparseLabelWiseStatisticVector();

            /**
             * An iterator that provides read-only access to the elements in the vector.
             */
            typedef ConstIterator const_iterator;

            /**
             * Returns a `const_iterator` to the beginning of the vector.
             *
             * @return A `const_iterator` to the beginning
             */
            const_iterator cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the vector.
             *
             * @return A `const_iterator` to the end
             */
            const_iterator cend() const;

            /**
             * Returns the number of elements in the vector.
             *
             * @return The number of elements in the vector
             */
            uint32 getNumElements() const;

            /**
             * Sets all gradients and Hessians in the vector to zero.
             */
            void clear();

            /**
             * Adds all gradients and Hessians in another vector to this vector.
             *
             * @param vector A reference to an object of type `SparseLabelWiseStatisticVector` that stores the gradients
             *               and Hessians to be added to this vector
             */
            void add(const SparseLabelWiseStatisticVector& vector);

            /**
             * Adds all gradients and Hessians in a single row of a `SparseLabelWiseStatisticConstView` to this vector.
             *
             * @param view  A reference to an object of type `SparseLabelWiseStatisticConstView` that stores the
             *              gradients and Hessians to be added to this vector
             * @param row   The index of the row to be added to this vector
             */
            void add(const SparseLabelWiseStatisticConstView& view, uint32 row);

            /**
             * Adds all gradients and Hessians in a single row of a `SparseLabelWiseStatisticConstView` to this vector.
             * The gradients and Hessians to be added are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `SparseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void add(const SparseLabelWiseStatisticConstView& view, uint32 row, float64 weight);

            /**
             * Removes all gradients and Hessians in a single row of a `SparseLabelWiseStatisticConstView` from this
             * vector.
             *
             * @param view  A reference to an object of type `SparseLabelWiseStatisticConstView` that stores the
             *              gradients and Hessians to be removed from this vector
             * @param row   The index of the row to be removed from this vector
             */
            void remove(const SparseLabelWiseStatisticConstView& view, uint32 row);

            /**
             * Removes all gradients and Hessians in a single row of a `SparseLabelWiseStatisticConstView` from this
             * vector. The gradients and Hessians to be added are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `SparseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be removed from this vector
             * @param row       The index of the row to be removed from this vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void remove(const SparseLabelWiseStatisticConstView& view, uint32 row, float64 weight);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseLabelWiseStatisticConstView`, whose
             * positions are given as a `CompleteIndexVector`, to this vector.
             *
             * @param view      A reference to an object of type `SparseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             */
            void addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                             const CompleteIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseLabelWiseStatisticConstView`, whose
             * positions are given as a `PartialIndexVector`, to this vector.
             *
             * @param view      A reference to an object of type `SparseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             */
            void addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                             const PartialIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseLabelWiseStatisticConstView`, whose
             * positions are given as a `CompleteIndexVector`, to this vector. The gradients and Hessians to be added
             * are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `SparseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                             const CompleteIndexVector& indices, float64 weight);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseLabelWiseStatisticConstView`, whose
             * positions are given as a `PartialIndexVector`, to this vector. The gradients and Hessians to be added are
             * multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `SparseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                             const PartialIndexVector& indices, float64 weight);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseLabelWiseHistogramConstView`, whose
             * positions are given as a `CompleteIndexVector`, to this vector.
             *
             * @param view      A reference to an object of type `SparseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             */
            void addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                             const CompleteIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseLabelWiseHistogramConstView`, whose
             * positions are given as a `PartialIndexVector`, to this vector.
             *
             * @param view      A reference to an object of type `SparseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             */
            void addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                             const PartialIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseLabelWiseHistogramConstView`, whose
             * positions are given as a `CompleteIndexVector`, to this vector. The gradients and Hessians to be added
             * are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `SparseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                             const CompleteIndexVector& indices, float64 weight);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseLabelWiseHistogramConstView`, whose
             * positions are given as a `PartialIndexVector`, to this vector. The gradients and Hessians to be added are
             * multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `SparseLabelWiseHistogramConstView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                             const PartialIndexVector& indices, float64 weight);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `CompleteIndexVector`.
             *
             * @param first         A reference to an object of type `SparseLabelWiseStatisticVector` that stores the
             *                      gradients and Hessians in the first vector
             * @param firstIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `SparseLabelWiseStatisticVector` that stores the
             *                      gradients and Hessians in the second vector
             */
            void difference(const SparseLabelWiseStatisticVector& first, const CompleteIndexVector& firstIndices,
                            const SparseLabelWiseStatisticVector& second);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `PartialIndexVector`.
             *
             * @param first         A reference to an object of type `SparseLabelWiseStatisticVector` that stores the
             *                      gradients and Hessians in the first vector
             * @param firstIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `SparseLabelWiseStatisticVector` that stores the
             *                      gradients and Hessians in the second vector
             */
            void difference(const SparseLabelWiseStatisticVector& first, const PartialIndexVector& firstIndices,
                            const SparseLabelWiseStatisticVector& second);
    };

}
