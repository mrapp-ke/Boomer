/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"

/**
 * A triple that consists of three values of the same type.
 *
 * @tparam T The type of the values
 */
template<typename T>
struct Triple final {
    public:

        Triple() {}

        /**
         * @param f The first value
         * @param s The second value
         * @param t The third value
         */
        Triple(T f, T s, T t) : first(f), second(s), third(t) {}

        /**
         * The first value.
         */
        T first;

        /**
         * The second value.
         */
        T second;

        /**
         * The third value.
         */
        T third;

        /**
         * Assigns a specific value to the first, second and third value of this triple.
         *
         * @param rhs   A reference to the value to be assigned
         * @return      A reference to the modified triple
         */
        Triple<T>& operator=(const T& rhs) {
            first = rhs;
            second = rhs;
            third = rhs;
            return *this;
        }

        /**
         * Adds a specific value to the first, second and third value of this triple.
         *
         * @param rhs   A reference to the value to be added
         * @return      A reference to the modified triple
         */
        Triple<T>& operator+=(const T& rhs) {
            first += rhs;
            second += rhs;
            third += rhs;
            return *this;
        }

        /**
         * Creates and returns a new triple that results from adding a specific value to the first, second and third
         * value of an existing triple.
         *
         * @param lhs   The original triple
         * @param rhs   A reference to the value to be added
         * @return      The triple that has been created
         */
        friend Triple<T> operator+(Triple<T> lhs, const T& rhs) {
            lhs += rhs;
            return lhs;
        }

        /**
         * Adds the first, second and third value of a given triple to the first, second and third value of this triple,
         * respectively,
         *
         * @param rhs   A reference to the triple, whose first, second and third value should be added
         * @return      A reference to the modified triple
         */
        Triple<T>& operator+=(const Triple<T>& rhs) {
            first += rhs.first;
            second += rhs.second;
            third += rhs.third;
            return *this;
        }

        /**
         * Creates and returns a new triple that results from adding the first, second and third value of a specific
         * triple to the first, second and third value of an existing triple, respectively.
         *
         * @param lhs   The original triple
         * @param rhs   A reference to the triple, whose first, second and third value should be added
         * @return      The triple that has been created
         */
        friend Triple<T> operator+(Triple<T> lhs, const Triple<T>& rhs) {
            lhs += rhs;
            return lhs;
        }

        /**
         * Subtracts a specific value from the first, second and third value of this triple.
         *
         * @param rhs   A reference to the value to be subtracted
         * @return      A reference to the modified triple
         */
        Triple<T>& operator-=(const T& rhs) {
            first -= rhs;
            second -= rhs;
            third -= rhs;
            return *this;
        }

        /**
         * Creates and returns a new triple that results from subtracting a specific value from the first, second and
         * third value of an existing triple, respectively.
         *
         * @param lhs   The original triple
         * @param rhs   A reference to the value to be subtracted
         * @return      The triple that has been created
         */
        friend Triple<T> operator-(Triple<T> lhs, const T& rhs) {
            lhs -= rhs;
            return lhs;
        }

        /**
         * Subtracts the first, second and third value of a given triple from the first, second and third value of this
         * triple, respectively.
         *
         * @param rhs   A reference to the triple, whose first, second and third value should be subtracted
         * @return      A reference to the modified triple
         */
        Triple<T>& operator-=(const Triple<T>& rhs) {
            first -= rhs.first;
            second -= rhs.second;
            third -= rhs.third;
            return *this;
        }

        /**
         * Creates and returns a new triple that results from subtracting the first, second and third value of a
         * specific triple from the first, second and third value of an existing triple, respectively.
         *
         * @param lhs   The original triple
         * @param rhs   A reference to the value to be subtracted
         * @return      The triple that has been created
         */
        friend Triple<T> operator-(Triple<T> lhs, const Triple<T>& rhs) {
            lhs -= rhs;
            return lhs;
        }

        /**
         * Multiplies the first, second and third value of this triple with a specific value.
         *
         * @param rhs   A reference to the value to be multiplied by
         * @return      A reference to the modified triple
         */
        Triple<T>& operator*=(const T& rhs) {
            first *= rhs;
            second *= rhs;
            third *= rhs;
            return *this;
        }

        /**
         * Creates and returns a new triple that results from multiplying the first, second and third value of an
         * existing triple with a specific value.
         *
         * @param lhs   The original triple
         * @param rhs   A reference to the value to be multiplied by
         * @return      The triple that has been created
         */
        friend Triple<T> operator*(Triple<T> lhs, const T& rhs) {
            lhs *= rhs;
            return lhs;
        }
};
