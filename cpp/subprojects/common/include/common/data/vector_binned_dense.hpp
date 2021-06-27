/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <iterator>


/**
 * An one-dimensional vector that provides random access to a fixed number of elements, corresponding to bins, stored in
 * a C-contiguous array.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<class T>
class DenseBinnedVector {

    private:

        uint32* binIndices_;

        T* array_;

        uint32 numElements_;

        uint32 numBins_;

        uint32 maxBinCapacity_;

    public:

        /**
         * Allows to iterate all elements in the vector.
         */
        class Iterator final {

            private:

                const DenseBinnedVector<T>& vector_;

                uint32 index_;

            public:

                /**
                 * @param vector    A reference to the vector that stores the elements
                 * @param index     The index to start at
                 */
                Iterator(const DenseBinnedVector<T>& vector, uint32 index);

                /**
                 * The type that is used to represent the difference between two iterators.
                 */
                typedef int difference_type;

                /**
                 * The type of the elements, the iterator provides access to.
                 */
                typedef T value_type;

                /**
                 * The type of a pointer to an element, the iterator provides access to.
                 */
                typedef T* pointer;

                /**
                 * The type of a reference to an element, the iterator provides access to.
                 */
                typedef T reference;

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
                Iterator& operator++();

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator to the next element
                 */
                Iterator& operator++(int n);

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator to the previous element
                 */
                Iterator& operator--();

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator to the previous element
                 */
                Iterator& operator--(int n);

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators refer to the same element, false otherwise
                 */
                bool operator!=(const Iterator& rhs) const;

                /**
                 * Returns the difference between this iterator and another one.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      The difference between the iterators
                 */
                difference_type operator-(const Iterator& rhs) const;

        };

        /**
         * @param numElements   The number of elements in the vector
         * @param numBins       The number of bins
         */
        DenseBinnedVector(uint32 numElements, uint32 numBins);

        virtual ~DenseBinnedVector();

        /**
         * An iterator that provides access to the indices that correspond to individual bins and allows to modify them.
         */
        typedef uint32* index_binned_iterator;

        /**
         * An iterator that provides read-only access to the indices that correspond to individual bins.
         */
        typedef const uint32* index_binned_const_iterator;

        /**
         * An iterator that provides access to the elements that correspond to individual bins and allows to modify
         * them.
         */
        typedef T* binned_iterator;

        /**
         * An iterator that provides read-only access to the elements that correspond to individual bins.
         */
        typedef const T* binned_const_iterator;

        /**
         * An iterator that provides read-only access to the elements in the vector.
         */
        typedef Iterator const_iterator;

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

        /**
         * Returns the value of the element at a specific position.
         *
         * @param pos   The position of the element. Must be in [0, getNumElements())
         * @return      The value of the given element
         */
        T getValue(uint32 pos) const;

};
