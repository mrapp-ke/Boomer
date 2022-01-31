/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/model/condition.hpp"
#include "common/model/body_conjunctive.hpp"
#include <memory>
#include <list>
#include <array>


/**
 * A list that stores conditions in the order they have been learned.
 */
class ConditionList final {

    private:

        std::list<Condition> list_;

        std::array<uint32, 4> numConditionsPerComparator_ = {0, 0, 0, 0};

    public:

        /**
         * The type that is used to store the size of the list.
         */
        typedef std::list<Condition>::size_type size_type;

        /**
         * An iterator that provides read-only access to the conditions in the list.
         */
        typedef std::list<Condition>::const_iterator const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the list.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the list.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Returns how many conditions are contained by the list in total.
         *
         * @return The number of conditions that are contained by the list
         */
        size_type getNumConditions() const;

        /**
         * Adds a new condition to the end of the list.
         *
         * @param condition A reference to an object of type `Condition` that should be added
         */
        void addCondition(const Condition& condition);

        /**
         * Removes the last condition from the list.
         */
        void removeLast();

        /**
         * Creates and returns a new object of type `ConjunctiveBody` from the conditions that contained by this list.
         *
         * @return An unique pointer to an object of type `ConjunctiveBody` that has been created
         */
        std::unique_ptr<ConjunctiveBody> createConjunctiveBody() const;

};
