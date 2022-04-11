/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * An one-dimension vector that stores binary data in a space-efficient way.
 */
class BitVector final {

    private:

        uint32 numElements_;

        uint32* array_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        BitVector(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        BitVector(uint32 numElements, bool init);

        ~BitVector();

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
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const;

        /**
         * Sets the values of all elements to zero.
         */
        void clear();

};
