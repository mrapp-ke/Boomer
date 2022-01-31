/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * A tuple that consists of an index and a value.
 *
 * @tparam T The type of the value
 */
template<typename T>
struct IndexedValue {

    /**
     * Allows to compare two objects of type `IndexedValue` according to the following strict weak ordering: If the
     * value of the first object is smaller, it goes before the second one. If the values of both objects are equal and
     * the index of the first object is smaller, it goes before the second one. Otherwise, the first object goes after
     * the second one.
     */
    struct Compare {

        /**
         * Returns whether the a given object of type `IndexedValue` should go before a second one.
         *
         * @param lhs   A reference to a first object of type `IndexedValue`
         * @param rhs   A reference to a second object of type `IndexedValue`
         * @return      True, if the first object should go before the second one, false otherwise
         */
        inline bool operator()(const IndexedValue<T>& lhs, const IndexedValue<T>& rhs) const {
            return lhs.value < rhs.value || (lhs.value == rhs.value && lhs.index < rhs.index);
        }

    };

    IndexedValue() { };

    /**
     * @param i The index
     * @param v The value
     */
    IndexedValue(uint32 i, T v) : index(i), value(v) { };

    /**
     * The index.
     */
    uint32 index;

    /**
     * The value.
     */
    T value;

};
