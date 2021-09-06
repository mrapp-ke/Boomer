/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/model/rule.hpp"
#include <forward_list>
#include <iterator>


/**
 * A model that stores several rules in a list.
 */
class RuleModel final {

    private:

        /**
         * Allows to iterate only the used rules.
         */
        class RuleConstIterator final {

            private:

                std::forward_list<Rule>::const_iterator iterator_;

                uint32 index_;

            public:

                /**
                 * @param list  A reference to the list that stores all available rules
                 * @param index The index to start at
                 */
                RuleConstIterator(const std::forward_list<Rule>& list, uint32 index);

                /**
                 * The type that is used to represent the difference between two iterators.
                 */
                typedef int difference_type;

                /**
                 * The type of the elements, the iterator provides access to.
                 */
                typedef const Rule value_type;

                /**
                 * The type of a pointer to an element, the iterator provides access to.
                 */
                typedef const Rule* pointer;

                /**
                 * The type of a reference to an element, the iterator provides access to.
                 */
                typedef const Rule& reference;

                /**
                 * The tag that specifies the capabilities of the iterator.
                 */
                typedef std::forward_iterator_tag iterator_category;

                /**
                 * Returns the element, the iterator currently refers to.
                 *
                 * @return The element, the iterator currently refers to
                 */
                reference operator*() const;

                /**
                 * Returns an iterator that refers to the next element.
                 *
                 * @return A reference to an iterator that refers to the next element
                 */
                RuleConstIterator& operator++();

                /**
                 * Returns an iterator that refers to the next element.
                 *
                 * @return A reference to an iterator that refers to the next element
                 */
                RuleConstIterator& operator++(int n);

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators do not refer to the same element, false otherwise
                 */
                bool operator!=(const RuleConstIterator& rhs) const;

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators refer to the same element, false otherwise
                 */
                bool operator==(const RuleConstIterator& rhs) const;

        };

        std::forward_list<Rule> list_;

        std::forward_list<Rule>::iterator it_;

        uint32 numRules_;

        uint32 numUsedRules_;

    public:

        RuleModel();

        /**
         * An iterator that provides read-only access to all rules.
         */
        typedef std::forward_list<Rule>::const_iterator const_iterator;

        /**
         * An iterator that provides read-only access to the used rules.
         */
        typedef RuleConstIterator used_const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the rules.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the rules.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Returns an `used_const_iterator` to the beginning of the used rules.
         *
         * @return An `used_const_iterator` to the beginning
         */
        used_const_iterator used_cbegin() const;

        /**
         * Returns an `used_const_iterator` to the end of the used rules.
         *
         * @return An `used_const_iterator` to the end
         */
        used_const_iterator used_cend() const;

        /**
         * Returns the total number of rules in the model.
         *
         * @return The number of rules
         */
        uint32 getNumRules() const;

        /**
         * Returns the number of used rules.
         *
         * @return The number of used rules
         */
        uint32 getNumUsedRules() const;

        /**
         * Sets the number of used rules.
         *
         * @param numUsedRules The number of used rules to be set or 0, if all rules are used
         */
        void setNumUsedRules(uint32 numUsedRules);

        /**
         * Creates a new rule from a given body and head and adds it to the model.
         *
         * @param bodyPtr An unique pointer to an object of type `IBody` that should be used as the body of the rule
         * @param headPtr An unique pointer to an object of type `IHead` that should be used as the head of the rule
         */
        void addRule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr);

        /**
         * Invokes some of the given visitor functions, depending on which ones are able to handle the bodies and heads
         * of the rules that are contained in this model.
         *
         * @param emptyBodyVisitor          The visitor function for handling objects of the type `EmptyBody`
         * @param conjunctiveBodyVisitor    The visitor function for handling objects of the type `ConjunctiveBody`
         * @param completeHeadVisitor       The visitor function for handling objects of the type `CompleteHead`
         * @param partialHeadVisitor        The visitor function for handling objects of the type `PartialHead`
         */
        void visit(IBody::EmptyBodyVisitor emptyBodyVisitor, IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                   IHead::CompleteHeadVisitor completeHeadVisitor, IHead::PartialHeadVisitor partialHeadVisitor) const;


        /**
         * Invokes some of the given visitor functions, depending on which ones are able to handle the bodies and heads
         * of the used rules that are contained in this model.
         *
         * @param emptyBodyVisitor          The visitor function for handling objects of the type `EmptyBody`
         * @param conjunctiveBodyVisitor    The visitor function for handling objects of the type `ConjunctiveBody`
         * @param completeHeadVisitor       The visitor function for handling objects of the type `CompleteHead`
         * @param partialHeadVisitor        The visitor function for handling objects of the type `PartialHead`
         */
        void visitUsed(IBody::EmptyBodyVisitor emptyBodyVisitor, IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                       IHead::CompleteHeadVisitor completeHeadVisitor,
                       IHead::PartialHeadVisitor partialHeadVisitor) const;

};
