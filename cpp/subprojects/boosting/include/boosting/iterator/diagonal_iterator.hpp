/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"

#include <iterator>

namespace boosting {

    /**
     * An iterator that provides read-only access to the elements that correspond to the diagonal of a C-contiguous
     * matrix.
     *
     * @tparam T The type of the elements that are stored in the matrix
     */
    template<typename T>
    class DiagonalConstIterator final {
        private:

            const T* ptr_;

            uint32 index_;

        public:

            /**
             * @param ptr   A pointer to a C-contiguous array of type `float64` that stores the elements of the matrix
             * @param index The index to start at
             */
            DiagonalConstIterator(const T* ptr, uint32 index);

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
            DiagonalConstIterator<T>& operator++();

            /**
             * Returns an iterator to the next element.
             *
             * @return A reference to an iterator to the next element
             */
            DiagonalConstIterator<T>& operator++(int n);

            /**
             * Returns an iterator to the previous element.
             *
             * @return A reference to an iterator to the previous element
             */
            DiagonalConstIterator<T>& operator--();

            /**
             * Returns an iterator to the previous element.
             *
             * @return A reference to an iterator to the previous element
             */
            DiagonalConstIterator<T>& operator--(int n);

            /**
             * Returns whether this iterator and another one refer to the same element.
             *
             * @param rhs   A reference to another iterator
             * @return      True, if the iterators do not refer to the same element, false otherwise
             */
            bool operator!=(const DiagonalConstIterator<T>& rhs) const;

            /**
             * Returns whether this iterator and another one refer to the same element.
             *
             * @param rhs   A reference to another iterator
             * @return      True, if the iterators refer to the same element, false otherwise
             */
            bool operator==(const DiagonalConstIterator<T>& rhs) const;

            /**
             * Returns the difference between this iterator and another one.
             *
             * @param rhs   A reference to another iterator
             * @return      The difference between the iterators
             */
            difference_type operator-(const DiagonalConstIterator<T>& rhs) const;
    };

}
