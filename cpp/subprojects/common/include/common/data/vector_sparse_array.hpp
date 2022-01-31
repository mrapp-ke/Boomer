/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/vector_dense.hpp"
#include "common/data/indexed_value.hpp"
#include <iterator>


/**
 * An one-dimensional sparse vector that stores a fixed number of elements, consisting of an index and a value, in a
 * C-contiguous array.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<typename T>
class SparseArrayVector final : public DenseVector<IndexedValue<T>> {

    private:

        /**
         * An iterator that provides random read-only access to the indices in a `SparseArrayVector`.
         */
        class IndexConstIterator final {

            private:

                typename VectorConstView<IndexedValue<T>>::const_iterator iterator_;

            public:

                /**
                 * @param iterator An iterator that provides access to the elements in the `SparseArrayVector`
                 */
                IndexConstIterator(typename VectorConstView<IndexedValue<T>>::const_iterator iterator);

                /**
                 * The type that is used to represent the difference between two iterators.
                 */
                typedef int difference_type;

                /**
                 * The type of the elements, the iterator provides access to.
                 */
                typedef uint32 value_type;

                /**
                 * The type of a pointer to an element, the iterator provides access to.
                 */
                typedef const uint32* pointer;

                /**
                 * The type of a reference to an element, the iterator provides access to.
                 */
                typedef const uint32& reference;

                /**
                 * The tag that specifies the capabilities of the iterator.
                 */
                typedef std::random_access_iterator_tag iterator_category;

                /**
                 * Returns the element at a specific index.
                 * @param index The index of the element to be returned
                 * @return      The element at the given index
                 */
                reference operator[](uint32 index) const;

                /**
                 * Returns the element, the iterator currently refers to.
                 *
                 * @return The element, the iterator currently refers to
                 */
                reference operator*() const;

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator that refers to the next element
                 */
                IndexConstIterator& operator++();

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator that refers to the next element
                 */
                IndexConstIterator& operator++(int n);

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator that refers to the previous element
                 */
                IndexConstIterator& operator--();

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator that refers to the previous element
                 */
                IndexConstIterator& operator--(int n);

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators do not refer to the same element, false otherwise
                 */
                bool operator!=(const IndexConstIterator& rhs) const;

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators refer to the same element, false otherwise
                 */
                bool operator==(const IndexConstIterator& rhs) const;

                /**
                 * Returns the difference between this iterator and another one.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      The difference between the iterators
                 */
                difference_type operator-(const IndexConstIterator& rhs) const;

        };

        /**
         * An iterator that provides random access to the indices in a `SparseArrayVector` and allows to modify them.
         */
        class IndexIterator final {

            private:

                typename VectorView<IndexedValue<T>>::iterator iterator_;

            public:

                /**
                 * @param iterator An iterator that provides access to the elements in the `SparseArrayVector`
                 */
                IndexIterator(typename VectorView<IndexedValue<T>>::iterator iterator);

                /**
                 * The type that is used to represent the difference between two iterators.
                 */
                typedef int difference_type;

                /**
                 * The type of the elements, the iterator provides access to.
                 */
                typedef uint32 value_type;

                /**
                 * The type of a pointer to an element, the iterator provides access to.
                 */
                typedef uint32* pointer;

                /**
                 * The type of a reference to an element, the iterator provides access to.
                 */
                typedef uint32& reference;

                /**
                 * The tag that specifies the capabilities of the iterator.
                 */
                typedef std::random_access_iterator_tag iterator_category;

                /**
                 * Returns the element at a specific index.
                 * @param index The index of the element to be returned
                 * @return      The element at the given index
                 */
                reference operator[](uint32 index) const;

                /**
                 * Returns the element, the iterator currently refers to.
                 *
                 * @return The element, the iterator currently refers to
                 */
                reference operator*() const;

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator that refers to the next element
                 */
                IndexIterator& operator++();

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator that refers to the next element
                 */
                IndexIterator& operator++(int n);

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator that refers to the previous element
                 */
                IndexIterator& operator--();

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator that refers to the previous element
                 */
                IndexIterator& operator--(int n);

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators do not refer to the same element, false otherwise
                 */
                bool operator!=(const IndexIterator& rhs) const;

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators refer to the same element, false otherwise
                 */
                bool operator==(const IndexIterator& rhs) const;

                /**
                 * Returns the difference between this iterator and another one.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      The difference between the iterators
                 */
                difference_type operator-(const IndexIterator& rhs) const;

        };

        /**
         * An iterator that provides random access to the values in a `SparseArrayVector` and allows to modify them.
         */
        class ValueConstIterator final {

            private:

                typename VectorConstView<IndexedValue<T>>::const_iterator iterator_;

            public:

                /**
                 * @param iterator An iterator that provides access to the elements in the `SparseArrayVector`
                 */
                ValueConstIterator(typename VectorConstView<IndexedValue<T>>::const_iterator iterator);

                /**
                 * The type that is used to represent the difference between two iterators.
                 */
                typedef int difference_type;

                /**
                 * The type of the elements, the iterator provides access to.
                 */
                typedef const T value_type;

                /**
                 * The type of a pointer to an element, the iterator provides access to.
                 */
                typedef const T* pointer;

                /**
                 * The type of a reference to an element, the iterator provides access to.
                 */
                typedef const T& reference;

                /**
                 * The tag that specifies the capabilities of the iterator.
                 */
                typedef std::random_access_iterator_tag iterator_category;

                /**
                 * Returns the element at a specific index.
                 * @param index The index of the element to be returned
                 * @return      The element at the given index
                 */
                reference operator[](uint32 index) const;

                /**
                 * Returns the element, the iterator currently refers to.
                 *
                 * @return The element, the iterator currently refers to
                 */
                reference operator*() const;

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator that refers to the next element
                 */
                ValueConstIterator& operator++();

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator that refers to the next element
                 */
                ValueConstIterator& operator++(int n);

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator that refers to the previous element
                 */
                ValueConstIterator& operator--();

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator that refers to the previous element
                 */
                ValueConstIterator& operator--(int n);

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators do not refer to the same element, false otherwise
                 */
                bool operator!=(const ValueConstIterator& rhs) const;

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators refer to the same element, false otherwise
                 */
                bool operator==(const ValueConstIterator& rhs) const;

                /**
                 * Returns the difference between this iterator and another one.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      The difference between the iterators
                 */
                difference_type operator-(const ValueConstIterator& rhs) const;

        };

        /**
         * An iterator that provides random access to the values in a `SparseArrayVector` and allows to modify them.
         */
        class ValueIterator final {

            private:

                typename VectorView<IndexedValue<T>>::iterator iterator_;

            public:

                /**
                 * @param iterator An iterator that provides access to the elements in the `SparseArrayVector`
                 */
                ValueIterator(typename VectorView<IndexedValue<T>>::iterator iterator);

                /**
                 * The type that is used to represent the difference between two iterators.
                 */
                typedef int difference_type;

                /**
                 * The type of the elements, the iterator provides access to.
                 */
                typedef T value_type;

                /**
                 * The type of a pointer to an element, the iterator provides access to.
                 */
                typedef T* pointer;

                /**
                 * The type of a reference to an element, the iterator provides access to.
                 */
                typedef T& reference;

                /**
                 * The tag that specifies the capabilities of the iterator.
                 */
                typedef std::random_access_iterator_tag iterator_category;

                /**
                 * Returns the element at a specific index.
                 * @param index The index of the element to be returned
                 * @return      The element at the given index
                 */
                reference operator[](uint32 index) const;

                /**
                 * Returns the element, the iterator currently refers to.
                 *
                 * @return The element, the iterator currently refers to
                 */
                reference operator*() const;

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator that refers to the next element
                 */
                ValueIterator& operator++();

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator that refers to the next element
                 */
                ValueIterator& operator++(int n);

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator that refers to the previous element
                 */
                ValueIterator& operator--();

                /**
                 * Returns an iterator to the previous element.
                 *
                 * @return A reference to an iterator that refers to the previous element
                 */
                ValueIterator& operator--(int n);

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators do not refer to the same element, false otherwise
                 */
                bool operator!=(const ValueIterator& rhs) const;

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators refer to the same element, false otherwise
                 */
                bool operator==(const ValueIterator& rhs) const;

                /**
                 * Returns the difference between this iterator and another one.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      The difference between the iterators
                 */
                difference_type operator-(const ValueIterator& rhs) const;

        };

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        SparseArrayVector(uint32 numElements);

        /**
         * An iterator that provides access to the indices in the vector and allows to modify them.
         */
        typedef IndexIterator index_iterator;

        /**
         * An iterator that provides read-only access to the indices in the vector.
         */
        typedef IndexConstIterator index_const_iterator;

        /**
         * An iterator that provides access to the values in the vector and allows to modify them.
         */
        typedef ValueIterator value_iterator;

        /**
         * An iterator that provides read-only access to the values in the vector.
         */
        typedef ValueConstIterator value_const_iterator;

        /**
         * Returns an `index_iterator` to the beginning of the indices in the vector.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator indices_begin();

        /**
         * Returns an `index_iterator` to the end of the indices in the vector.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator indices_end();

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
         * Returns a `value_iterator` to the beginning of the values in the vector.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin();

        /**
         * Returns a `value_iterator` to the end of the values in the vector.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end();

        /**
         * Returns a `value_const_iterator` to the beginning of the values in the vector.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const;

        /**
         * Returns a `value_const_iterator` to the end of the values in the vector.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const;

        /**
         * Sorts the elements in the vector in ascending order based on their values.
         */
        void sortByValues();

};
