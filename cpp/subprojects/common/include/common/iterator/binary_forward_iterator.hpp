/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"

#include <iterator>

/**
 * An iterator adaptor that adapts an iterator, which provides access to a fixed number of indices in increasing order,
 * such that it acts as a forward iterator that returns a boolean value for each possible index, indicating whether the
 * respective index is present in the original iterator or not.
 *
 * @tparam Iterator The type of the iterator to be adapted
 */
template<typename Iterator>
class BinaryForwardIterator final {
    private:

        Iterator iterator_;

        Iterator end_;

        uint32 index_;

        uint32 iteratorIndex_;

    public:

        /**
         * @param begin An iterator to the beginning of the indices
         * @param end   An iterator to the end of the indices
         * @param index The index to start at
         */
        BinaryForwardIterator(Iterator begin, Iterator end, uint32 index)
            : iterator_(begin), end_(end), index_(index), iteratorIndex_(iterator_ != end_ ? *iterator_ : 0) {}

        /**
         * The type that is used to represent the difference between two iterators.
         */
        typedef int difference_type;

        /**
         * The type of the elements, the iterator provides access to.
         */
        typedef bool value_type;

        /**
         * The type of a pointer to an element, the iterator provides access to.
         */
        typedef const bool* pointer;

        /**
         * The type of a reference to an element, the iterator provides access to.
         */
        typedef bool reference;

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
            return iterator_ != end_ && iteratorIndex_ == index_;
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        BinaryForwardIterator<Iterator>& operator++() {
            ++index_;

            if (iterator_ != end_ && iteratorIndex_ < index_) {
                iterator_++;

                if (iterator_ != end_) {
                    iteratorIndex_ = *iterator_;
                }
            }

            return *this;
        }

        /**
         * Returns an iterator to the next element.
         *
         * @return A reference to an iterator that refers to the next element
         */
        BinaryForwardIterator<Iterator>& operator++(int n) {
            index_++;

            if (iterator_ != end_ && iteratorIndex_ < index_) {
                iterator_++;

                if (iterator_ != end_) {
                    iteratorIndex_ = *iterator_;
                }
            }

            return *this;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators do not refer to the same element, false otherwise
         */
        bool operator!=(const BinaryForwardIterator<Iterator>& rhs) const {
            return index_ != rhs.index_;
        }

        /**
         * Returns whether this iterator and another one refer to the same element.
         *
         * @param rhs   A reference to another iterator
         * @return      True, if the iterators refer to the same element, false otherwise
         */
        bool operator==(const BinaryForwardIterator<Iterator>& rhs) const {
            return index_ == rhs.index_;
        }
};

/**
 * Creates and returns a new `BinaryForwardIterator`.
 *
 * @tparam Iterator The type of the iterator to be adapted
 * @param begin     An iterator to the beginning of the indices
 * @param end       An iterator to the end of the indices
 * @param index     The index to start at
 * @return          A `BinaryForwardIterator` that has been created
 */
template<typename Iterator>
static inline BinaryForwardIterator<Iterator> make_binary_forward_iterator(Iterator begin, Iterator end,
                                                                           uint32 index = 0) {
    return BinaryForwardIterator<Iterator>(begin, end, index);
}
