/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"

#include <vector>

/**
 * A two-dimensional matrix that provides row-wise access to data stored in a lists of lists.
 *
 * @tparam T The type of the data that is stored by the matrix
 */
template<typename T>
class ListOfLists final {
    private:

        const uint32 numRows_;

        std::vector<T>* array_;

    public:

        /**
         * @param numRows The number of rows in the matrix
         */
        ListOfLists(uint32 numRows);

        ~ListOfLists();

        /**
         * Provides access to a row and allows to modify its elements.
         */
        typedef typename std::vector<T>& row;

        /**
         * Provides read-only access to a row.
         */
        typedef const typename std::vector<T>& const_row;

        /**
         * An iterator that provides access to the elements at a row and allows to modify them.
         */
        typedef typename std::vector<T>::iterator iterator;

        /**
         * An iterator that provides read-only access to the elements at a row.
         */
        typedef typename std::vector<T>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the beginning
         */
        iterator begin(uint32 row);

        /**
         * Returns an `iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the end
         */
        iterator end(uint32 row);

        /**
         * Returns a `const_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the beginning
         */
        const_iterator cbegin(uint32 row) const;

        /**
         * Returns a `const_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the end
         */
        const_iterator cend(uint32 row) const;

        /**
         * Provides access to a specific row and allows to modify its elements.
         *
         * @param row   The index of the row
         * @return      A `row`
         */
        row operator[](uint32 row);

        /**
         * Provides read-only access to a specific row.
         *
         * @param row   The index of the row
         * @return      A `const_row`
         */
        const_row operator[](uint32 row) const;

        /**
         * Returns the number of rows in the matrix.
         *
         * @return The number of rows
         */
        uint32 getNumRows() const;

        /**
         * Sets the values of all elements to zero.
         */
        void clear();
};
