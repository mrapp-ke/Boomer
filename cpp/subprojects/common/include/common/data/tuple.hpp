/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * A tuple that consists of two values of the same type.
 *
 * @tparam T The type of the values
 */
template<typename T>
struct Tuple {

    Tuple() { };

    /**
     * @param f The first value
     * @param s The second value
     */
    Tuple(T f, T s) : first(f), second(s) { };

    /**
     * The first value.
     */
    T first;

    /**
     * The second value.
     */
    T second;

    /**
     * Assigns a specific value to the first and second value of this tuple.
     *
     * @param rhs   A reference to the value to be assigned
     * @return      A reference to the modified tuple
     */
    Tuple<T>& operator=(const T& rhs) {
        first = rhs;
        second = rhs;
        return *this;
    }

    /**
     * Adds a specific value to the first and second value of this tuple.
     *
     * @param rhs   A reference to the value to be added
     * @return      A reference to the modified tuple
     */
    Tuple<T>& operator+=(const T& rhs) {
        first += rhs;
        second += rhs;
        return *this;
    }

    /**
     * Creates and returns a new tuple that results from adding a specific value to the first and second value of an
     * existing tuple.
     *
     * @param lhs   The original tuple
     * @param rhs   A reference to the value to be added
     * @return      The tuple that has been created
     */
    friend Tuple<T> operator+(Tuple<T> lhs, const T& rhs) {
        lhs += rhs;
        return lhs;
    }

    /**
     * Adds the first and second value of a given tuple to the first and second value of this tuple, respectively,
     *
     * @param rhs   A reference to the tuple, whose first and second value should be added
     * @return      A reference to the modified tuple
     */
    Tuple<T>& operator+=(const Tuple<T>& rhs) {
        first += rhs.first;
        second += rhs.second;
        return *this;
    }

    /**
     * Creates and returns a new tuple that results from adding the first and second value of a specific tuple to the
     * first and second value of an existing tuple, respectively.
     *
     * @param lhs   The original tuple
     * @param rhs   A reference to the tuple, whose first and second value should be added
     * @return      The tuple that has been created
     */
    friend Tuple<T> operator+(Tuple<T> lhs, const Tuple<T>& rhs) {
        lhs += rhs;
        return lhs;
    }

    /**
     * Subtracts a specific value from the first and second value of this tuple.
     *
     * @param rhs   A reference to the value to be subtracted
     * @return      A reference to the modified tuple
     */
    Tuple<T>& operator-=(const T& rhs) {
        first -= rhs;
        second -= rhs;
        return *this;
    }

    /**
     * Creates and returns a new tuple that results from subtracting a specific value from the first and second value of
     * an existing tuple, respectively.
     *
     * @param lhs   The original tuple
     * @param rhs   A reference to the value to be subtracted
     * @return      The tuple that has been created
     */
    friend Tuple<T> operator-(Tuple<T> lhs, const T& rhs) {
        lhs -= rhs;
        return lhs;
    }

    /**
     * Subtracts the first and second value of a given tuple from the first and second value of this tuple,
     * respectively.
     *
     * @param rhs   A reference to the tuple, whose first and second value should be subtracted
     * @return      A reference to the modified tuple
     */
    Tuple<T>& operator-=(const Tuple<T>& rhs) {
        first -= rhs.first;
        second -= rhs.second;
        return *this;
    }

    /**
     * Creates and returns a new tuple that results from subtracting the first and second value of a specific tuple from
     * the first and second value of an existing tuple, respectively.
     *
     * @param lhs   The original tuple
     * @param rhs   A reference to the value to be subtracted
     * @return      The tuple that has been created
     */
    friend Tuple<T> operator-(Tuple<T> lhs, const Tuple<T>& rhs) {
        lhs -= rhs;
        return lhs;
    }

    /**
     * Multiplies the first and second value of this tuple with a specific value.
     *
     * @param rhs   A reference to the value to be multiplied by
     * @return      A reference to the modified tuple
     */
    Tuple<T>& operator*=(const T& rhs) {
        first *= rhs;
        second *= rhs;
        return *this;
    }

    /**
     * Creates and returns a new tuple that results from multiplying the first and second value of an existing tuple
     * with a specific value.
     *
     * @param lhs   The original tuple
     * @param rhs   A reference to the value to be multiplied by
     * @return      The tuple that has been created
     */
    friend Tuple<T> operator*(Tuple<T> lhs, const T& rhs) {
        lhs *= rhs;
        return lhs;
    }

};
