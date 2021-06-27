/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <utility>


/**
 * A ring buffer with fixed capacity.
 *
 * @tparam T The type of the values that are stored in the buffer
 */
template<class T>
class RingBuffer final {

    private:

        T* array_;

        uint32 capacity_;

        uint32 pos_;

        bool full_;

    public:

        /**
         * @param capacity The maximum capacity of the buffer. Must be at least 1
         */
        RingBuffer(uint32 capacity);

        ~RingBuffer();

        /**
         * An iterator that provides read-only access to the elements in the buffer.
         */
        typedef const T* const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the buffer.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the buffer.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Returns the maximum capacity of the buffer.
         *
         * @return The maximum capacity
         */
        uint32 getCapacity() const;

        /**
         * Returns the number of values in the buffer.
         *
         * @return The number of values
         */
        uint32 getNumElements() const;

        /**
         * Returns whether the maximum capacity of the buffer has been reached or not.
         *
         * @return True, if the maximum capacity has been reached, false otherwise
         */
        bool isFull() const;

        /**
         * Adds a new value to the buffer. If the maximum capacity of the buffer has been reached, the oldest value will
         * be overwritten.
         *
         * @param value The value to be added
         * @return      A `std::pair`, whose first value indicates whether a value has been overwritten or not. If a
         *              value has been overwritten, the pair's second value is set to the overwritten value, otherwise
         *              it is undefined
         */
        std::pair<bool, T> push(T value);

};
