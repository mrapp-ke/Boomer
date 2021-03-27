/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include "common/data/indexed_value.hpp"


/**
 * An one-dimensional sparse vector that stores a fixed number of elements, consisting of an index and a value, in a
 * C-contiguous array.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<class T>
class SparseArrayVector final {

    public:

        /**
         * The type of an element that is contained by the vector.
         */
        typedef IndexedValue<T> Entry;


    private:

        Entry* array_;

        uint32 numElements_;

        uint32 maxCapacity_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        SparseArrayVector(uint32 numElements);

        virtual ~SparseArrayVector();

        /**
         * An iterator that provides access to the elements in the vector and allows to modify them.
         */
        typedef Entry* iterator;

        /**
         * An iterator that provides read-only access to the elements in the vector.
         */
        typedef const Entry* const_iterator;

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
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const;

        /**
         * Sets the number of elements in the vector.
         *
         * @param numElements   The number of elements to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumElements(uint32 numElements, bool freeMemory);

        /**
         * Sorts the elements in the vector in ascending order based on their values.
         */
        void sortByValues();

};
