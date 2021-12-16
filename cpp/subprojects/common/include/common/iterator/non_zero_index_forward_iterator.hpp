/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include <iterator>
#include <type_traits>


/**
 * An iterator adaptor that adapts an iterator, which provides access to a fixed number of values, such that it acts as
 * a forward iterator that returns the indices of all non-zero values.
 *
 * @tparam T The type of the iterator to be adapted
 */
template<typename T>
class NonZeroIndexForwardIterator {

    private:

        T iterator_;

        T end_;

        uint32 index_;

    public:

        /**
         * @param begin An iterator to the beginning of the values
         * @param end   An iterator to the end of the values
         */
        NonZeroIndexForwardIterator(T begin, T end)
            : iterator_(begin), end_(end), index_(0) {
            for (; iterator_ != end_; iterator_++) {
                auto value = *iterator_;

                if (value != 0) {
                    break;
                }

                index_++;
            }
        }

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
        typedef std::forward_iterator_tag iterator_category;

        /**
         * Returns the element, the iterator currently refers to.
         *
         * @return The element, the iterator currently refers to
         */
        reference operator*() const {
            return index_;
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        NonZeroIndexForwardIterator<T>& operator++() {
            iterator_++;
            ++index_;

            for (; iterator_ != end_; iterator_++) {
                auto value = *iterator_;

                if (value != 0) {
                    break;
                }

                ++index_;
            }

            return *this;
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        NonZeroIndexForwardIterator<T>& operator++(int n) {
            iterator_++;
            index_++;

            for (; iterator_ != end_; iterator_++) {
                auto value = *iterator_;

                if (value != 0) {
                    break;
                }

                index_++;
            }

            return *this;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators do not refer to the same element, false otherwise
         */
        bool operator!=(const NonZeroIndexForwardIterator<T>& rhs) const {
            return iterator_ != rhs.iterator_;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators refer to the same element, false otherwise
         */
        bool operator==(const NonZeroIndexForwardIterator<T>& rhs) const {
            return iterator_ == rhs.iterator_;
        }

};

/**
 * Creates and returns a new `NonZeroIndexForwardIterator`.
 *
 * @param begin An iterator to the beginning of the values
 * @param end   An iterator to the end of the values
 * @return      A `NonZeroIndexForwardIterator` that has been created
 */
template<typename T>
static inline NonZeroIndexForwardIterator<T> make_non_zero_index_forward_iterator(T begin, T end) {
    return NonZeroIndexForwardIterator<T>(begin, end);
}
