/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/vector_sparse_list_binary.hpp"


/**
 * A two-dimensional matrix that provides row-wise access to binary values that are stored in the list of lists (LIL)
 * format.
 */
class BinaryLilMatrix final {

    public:

        /**
         * The type of a row in the matrix.
         */
        typedef BinarySparseListVector Row;

    private:

        uint32 numRows_;

        Row* array_;

    public:

        /**
         * @param numRows The number of rows in the matrix
         */
        BinaryLilMatrix(uint32 numRows);

        ~BinaryLilMatrix();

        /**
         * An iterator that provides access to the elements at a row and allows to modify them.
         */
        typedef typename BinarySparseListVector::iterator iterator;

        /**
         * An iterator that provides read-only access to the elements at a row.
         */
        typedef typename BinarySparseListVector::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the beginning
         */
        iterator row_begin(uint32 row);

        /**
         * Returns an `iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the end
         */
        iterator row_end(uint32 row);

        /**
         * Returns a `const_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the beginning
         */
        const_iterator row_cbegin(uint32 row) const;

        /**
         * Returns a `const_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the end
         */
        const_iterator row_cend(uint32 row) const;

        /**
         * Returns a reference to a specific row.
         *
         * @param row   The row
         * @return      A reference to the row
         */
        Row& getRow(uint32 row);

        /**
         * Returns a const reference to a specific row.
         *
         * @param row   The row
         * @return      A const reference to the row
         */
        const Row& getRow(uint32 row) const;

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
