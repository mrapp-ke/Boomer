/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <iterator>


/**
 * An iterator that provides random read-only access to the indices in a continuous range.
 */
class IndexIterator final {

    private:

        uint32 index_;

    public:

        IndexIterator();

        /**
         * @param index The index to start with
         */
        IndexIterator(uint32 index);

        /**
         * The type that is used to represent the difference between two iterators.
         */
        typedef int difference_type;

        /**
         * The type of the elements, the iterator provides access to.
         */
        typedef uint32 value_type;

        /**
         * The type of a pointer to an element, the iterator provides access to.
         */
        typedef const uint32* pointer;

        /**
         * The type of a reference to an element, the iterator provides access to.
         */
        typedef uint32 reference;

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
         * @return A reference to an iterator that refers to the next element
         */
        IndexIterator& operator++();

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        IndexIterator& operator++(int n);

        /**
         * Returns an iterator to the previous element.
         *
         * @return A reference to an iterator that refers to the previous element
         */
        IndexIterator& operator--();

        /**
         * Returns an iterator to the previous element.
         *
         * @return A reference to an iterator that refers to the previous element
         */
        IndexIterator& operator--(int n);

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators do not refer to the same element, false otherwise
         */
        bool operator!=(const IndexIterator& rhs) const;

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators refer to the same element, false otherwise
         */
        bool operator==(const IndexIterator& rhs) const;

        /**
         * Returns the difference between this iterator and another one.
         *
         * @param rhs   A reference to another iterator
         * @return      The difference between the iterators
         */
        difference_type operator-(const IndexIterator& rhs) const;

};
