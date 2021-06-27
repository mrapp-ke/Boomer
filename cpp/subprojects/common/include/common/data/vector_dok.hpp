/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <unordered_map>


/**
 * An one-dimensional sparse vector that stores data using the dictionary of keys (DOK) format.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<class T>
class DokVector final {

    private:

        std::unordered_map<uint32, T> data_;

        T sparseValue_;

    public:

        /**
         * @param sparseValue The value of sparse elements
         */
        DokVector(T sparseValue);

        /**
         * An iterator that provides access to the elements in the vector and allows to modify them.
         */
        typedef typename std::unordered_map<uint32, T>::iterator iterator;

        /**
         * An iterator that provides read-only access to the elements in the vector.
         */
        typedef typename std::unordered_map<uint32, T>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of the vector.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        iterator end();

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
         * Returns the value of the element at a specific position.
         *
         * @param pos   The position of the element. Must be in [0, getNumElements())
         * @return      The value of the given element
         */
        T getValue(uint32 pos) const;

        /**
         * Sets a value to the element at a specific position.
         *
         * @param pos   The position of the element. Must be in [0, getNumElements())
         * @param value The value to be set
         */
        void setValue(uint32 pos, T value);

        /**
         * Sets the values of all elements to zero.
         */
        void setAllToZero();

};
