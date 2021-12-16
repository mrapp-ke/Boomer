/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include <unordered_set>


/**
 * An one-dimensional sparse vector that stores binary data using the dictionary of keys (DOK) format.
 */
class BinaryDokVector final {

    private:

        std::unordered_set<uint32> data_;

    public:

        /**
         * An iterator that provides read-only access to the elements in the vector.
         */
        typedef std::unordered_set<uint32>::const_iterator index_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        /**
         * Returns the value of the element at a specific position.
         *
         * @param pos   The position of the element
         * @return      The value of the specified element
         */
        bool operator[](uint32 pos) const;

        /**
         * Sets a value to the element at a specific position.
         *
         * @param pos   The position of the element
         * @param value The value to be set
         */
        void set(uint32 pos, bool value);

        /**
         * Sets the values of all elements to zero.
         */
        void clear();

};
