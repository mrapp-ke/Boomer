/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <list>


/**
 * An one-dimensional sparse vector that stores binary data by including the indices of all non-zero elements in a
 * double-linked list.
 */
class BinarySparseListVector {

    private:

        std::list<uint32> indices_;

    public:

        /**
         * An iterator that provides read-only access to the indices in the vector.
         */
        typedef std::list<uint32>::const_iterator index_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices in the vector.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices in the vector.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const;

        /**
         * Sets a non-zero value to the element at a specific position.
         *
         * @param pos The position of the element. Must be in [0, getNumElements())
         */
        void setValue(uint32 pos);

        /**
         * Sets the values of all elements to zero.
         */
        void setAllToZero();

};
