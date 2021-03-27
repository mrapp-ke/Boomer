/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <unordered_set>
#include <utility>


/**
 * A two-dimensional sparse matrix that stores binary data using the dictionary of keys (DOK) format.
 */
class BinaryDokMatrix final {

    private:

        typedef std::pair<uint32, uint32> Entry;

        /**
         * Implements a hash function for elements of type `Entry`..
         */
        struct HashFunction {

            inline std::size_t operator()(const Entry &v) const {
                return (((uint64) v.first) << 32) | ((uint64) v.second);
            }

        };

        std::unordered_set<Entry, HashFunction> data_;

    public:

        /**
         * Returns the value of the element at a specific position.
         *
         * @param row   The row of the element. Must be in [0, getNumRows())
         * @param col   The column of the element. Must be in [0, getNumCols())
         * @return      The value of the given element
         */
        bool getValue(uint32 row, uint32 col) const;

        /**
         * Sets a non-zero value to the element at a specific position.
         *
         * @param row       The row of the element. Must be in [0, getNumRows())
         * @param column    The column of the element. Must be in [0, getNumCols())
         */
        void setValue(uint32 row, uint32 column);

        /**
         * Sets the values of all elements to zero.
         */
        void setAllToZero();

};
