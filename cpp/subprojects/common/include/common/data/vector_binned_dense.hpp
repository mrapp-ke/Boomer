/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/vector_dense.hpp"
#include <iterator>


/**
 * An one-dimensional vector that provides random access to a fixed number of elements, corresponding to bins, stored in
 * a C-contiguous array.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<typename T>
class DenseBinnedVector {

    private:

        DenseVector<uint32> binIndices_;

        DenseVector<T> values_;

    public:

        /**
         * An iterator that provides read-only access to the values of all elements in a `DenseBinnedVector`.
         */
        class ValueConstIterator final {

            private:

                DenseVector<uint32>::const_iterator binIndexIterator_;

                typename DenseVector<T>::const_iterator valueIterator_;

            public:

                /**
                 * @param binIndexIterator  An iterator to the bin indices of individual elements
                 * @param valueIterator     An iterator to the values of individual bins
                 */
                ValueConstIterator(DenseVector<uint32>::const_iterator binIndexIterator,
                                   typename DenseVector<T>::const_iterator valueIterator);

                /**
                 * The type that is used to represent the difference between two iterators.
                 */
                typedef int difference_type;

                /**
                 * The type of the elements, the iterator provides access to.
                 */
                typedef const T value_type;

                /**
                 * The type of a pointer to an element, the iterator provides access to.
                 */
                typedef const T* pointer;

                /**
                 * The type of a reference to an element, the iterator provides access to.
                 */
                typedef const T& reference;

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
                reference operator[](uint32 index) const;

                /**
                 * Returns the element, the iterator currently refers to.
                 *
                 * @return The element, the iterator currently refers to
                 */
                reference operator*() const;

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator to the next element
                 */
                ValueConstIterator& operator++();

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator to the next element
                 */
                ValueConstIterator& operator++(int n);

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator to the previous element
                 */
                ValueConstIterator& operator--();

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator to the previous element
                 */
                ValueConstIterator& operator--(int n);

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators do not refer to the same element, false otherwise
                 */
                bool operator!=(const ValueConstIterator& rhs) const;

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators refer to the same element, false otherwise
                 */
                bool operator==(const ValueConstIterator& rhs) const;

                /**
                 * Returns the difference between this iterator and another one.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      The difference between the iterators
                 */
                difference_type operator-(const ValueConstIterator& rhs) const;

        };

        /**
         * @param numElements   The number of elements in the vector
         * @param numBins       The number of bins
         */
        DenseBinnedVector(uint32 numElements, uint32 numBins);

        /**
         * An iterator that provides access to the indices that correspond to individual bins and allows to modify them.
         */
        typedef typename DenseVector<uint32>::iterator index_binned_iterator;

        /**
         * An iterator that provides read-only access to the indices that correspond to individual bins.
         */
        typedef typename DenseVector<uint32>::const_iterator index_binned_const_iterator;

        /**
         * An iterator that provides access to the elements that correspond to individual bins and allows to modify
         * them.
         */
        typedef typename DenseVector<T>::iterator binned_iterator;

        /**
         * An iterator that provides read-only access to the elements that correspond to individual bins.
         */
        typedef typename DenseVector<T>::const_iterator binned_const_iterator;

        /**
         * An iterator that provides read-only access to the elements in the vector.
         */
        typedef ValueConstIterator const_iterator;

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
         * Returns an `index_binned_iterator` to the beginning of the indices that correspond to individual bins.
         *
         * @return An `index_binned_iterator` to the beginning
         */
        index_binned_iterator indices_binned_begin();

        /**
         * Returns an `index_binned_iterator` to the end of the indices that correspond to individual bins.
         *
         * @return An `index_binned_iterator` to the end
         */
        index_binned_iterator indices_binned_end();

        /**
         * Returns an `index_binned_const_iterator` to the beginning of the indices that correspond to individual bins.
         *
         * @return An `index_binned_const_iterator` to the beginning
         */
        index_binned_const_iterator indices_binned_cbegin() const;

        /**
         * Returns an `index_binned_const_iterator` to the end of the indices that correspond to individual bins.
         *
         * @return An `index_binned_const_iterator` to the end
         */
        index_binned_const_iterator indices_binned_cend() const;

        /**
         * Returns a `binned_iterator` to the beginning of the elements that correspond to individual bins.
         *
         * @return A `binned_iterator` to the beginning
         */
        binned_iterator binned_begin();

        /**
         * Returns a `binned_iterator` to the end of the elements that correspond to individual bins.
         *
         * @return A `binned_iterator` to the end
         */
        binned_iterator binned_end();

        /**
         * Returns a `binned_const_iterator` to the beginning of the elements that correspond to individual bins.
         *
         * @return A `binned_const_iterator` to the beginning
         */
        binned_const_iterator binned_cbegin() const;

        /**
         * Returns a `binned_const_iterator` to the end of the elements that correspond to individual bins.
         *
         * @return A `binned_const_iterator` to the end
         */
        binned_const_iterator binned_cend() const;

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const;

        /**
         * Returns the number of bins.
         *
         * @return The number of bins
         */
        uint32 getNumBins() const;

        /**
         * Sets the number of bins.
         *
         * @param numBins       The number of bins to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumBins(uint32 numBins, bool freeMemory);

};
