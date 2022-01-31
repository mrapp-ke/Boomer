/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/model/rule_model.hpp"
#include "common/model/body.hpp"
#include "common/model/head.hpp"
#include <forward_list>
#include <iterator>


/**
 * Defines an interface for all rule-based models that store several rules in an ordered list.
 */
class MLRLCOMMON_API IRuleList : public IRuleModel {

    public:

        virtual ~IRuleList() override { };

        /**
         * Creates a new default rule from a given head and adds it to the end of the model.
         *
         * @param headPtr An unique pointer to an object of type `IHead` that should be used as the head of the rule
         */
        virtual void addDefaultRule(std::unique_ptr<IHead> headPtr) = 0;

        /**
         * Creates a new rule from a given body and head and adds it to the end of the model.
         *
         * @param bodyPtr An unique pointer to an object of type `IBody` that should be used as the body of the rule
         * @param headPtr An unique pointer to an object of type `IHead` that should be used as the head of the rule
         */
        virtual void addRule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr) = 0;

        /**
         * Returns whether the model contains a default rule or not.
         *
         * @return True, if the model contains a default rule, false otherwise
         */
        virtual bool containsDefaultRule() const = 0;

        /**
         * Invokes some of the given visitor functions, depending on which ones are able to handle the bodies and heads
         * of the rules that are contained in this model.
         *
         * @param emptyBodyVisitor          The visitor function for handling objects of the type `EmptyBody`
         * @param conjunctiveBodyVisitor    The visitor function for handling objects of the type `ConjunctiveBody`
         * @param completeHeadVisitor       The visitor function for handling objects of the type `CompleteHead`
         * @param partialHeadVisitor        The visitor function for handling objects of the type `PartialHead`
         */
        virtual void visit(IBody::EmptyBodyVisitor emptyBodyVisitor,
                           IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                           IHead::CompleteHeadVisitor completeHeadVisitor,
                           IHead::PartialHeadVisitor partialHeadVisitor) const = 0;


        /**
         * Invokes some of the given visitor functions, depending on which ones are able to handle the bodies and heads
         * of the used rules that are contained in this model.
         *
         * @param emptyBodyVisitor          The visitor function for handling objects of the type `EmptyBody`
         * @param conjunctiveBodyVisitor    The visitor function for handling objects of the type `ConjunctiveBody`
         * @param completeHeadVisitor       The visitor function for handling objects of the type `CompleteHead`
         * @param partialHeadVisitor        The visitor function for handling objects of the type `PartialHead`
         */
        virtual void visitUsed(IBody::EmptyBodyVisitor emptyBodyVisitor,
                               IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                               IHead::CompleteHeadVisitor completeHeadVisitor,
                               IHead::PartialHeadVisitor partialHeadVisitor) const = 0;

};

/**
 * An implementation of the type `IRuleList` that stores several rules in a single-linked list.
 */
class RuleList final : public IRuleList {

    public:

        /**
         * An implementation of the type `IRule` that stores unique pointers to the body and head of a rule.
         */
        class Rule final {

            private:

                std::unique_ptr<IBody> bodyPtr_;

                std::unique_ptr<IHead> headPtr_;

            public:

                /**
                 * @param bodyPtr   An unique pointer to an object of type `IBody` that represents the body of the rule
                 * @param headPtr   An unique pointer to an object of type `IHead` that represents the head of the rule
                 */
                Rule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr);

                /**
                 * Returns the body of the rule.
                 *
                 * @return A reference to an object of type `IBody` that represents the body of the rule
                 */
                const IBody& getBody() const;

                /**
                 * Returns the head of the rule.
                 *
                 * @return A reference to an object of type `IHead` that represents the head of the rule
                 */
                const IHead& getHead() const;

                /**
                 * Invokes some of the given visitor functions, depending on which ones are able to handle the rule's
                 * particular type of body and head.
                 *
                 * @param emptyBodyVisitor          The visitor function for handling objects of type `EmptyBody`
                 * @param conjunctiveBodyVisitor    The visitor function for handling objects of type `ConjunctiveBody`
                 * @param completeHeadVisitor       The visitor function for handling objects of type `CompleteHead`
                 * @param partialHeadVisitor        The visitor function for handling objects of type `PartialHead`
                 */
                void visit(IBody::EmptyBodyVisitor emptyBodyVisitor,
                           IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                           IHead::CompleteHeadVisitor completeHeadVisitor,
                           IHead::PartialHeadVisitor partialHeadVisitor) const;

        };

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

        bool containsDefaultRule_;

    public:

        RuleList();

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

        uint32 getNumRules() const override;

        uint32 getNumUsedRules() const override;

        void setNumUsedRules(uint32 numUsedRules) override;

        void addDefaultRule(std::unique_ptr<IHead> headPtr) override;

        void addRule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr) override;

        bool containsDefaultRule() const override;

        void visit(IBody::EmptyBodyVisitor emptyBodyVisitor,
                   IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                   IHead::CompleteHeadVisitor completeHeadVisitor,
                   IHead::PartialHeadVisitor partialHeadVisitor) const override;


        void visitUsed(IBody::EmptyBodyVisitor emptyBodyVisitor,
                       IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                       IHead::CompleteHeadVisitor completeHeadVisitor,
                       IHead::PartialHeadVisitor partialHeadVisitor) const override;

        std::unique_ptr<IClassificationPredictor> createClassificationPredictor(
            const IClassificationPredictorFactory& factory, const ILabelSpaceInfo& labelSpaceInfo) const override;

        std::unique_ptr<IRegressionPredictor> createRegressionPredictor(
            const IRegressionPredictorFactory& factory, const ILabelSpaceInfo& labelSpaceInfo) const override;

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
            const IProbabilityPredictorFactory& factory, const ILabelSpaceInfo& labelSpaceInfo) const override;

};

/**
 * Creates and returns a new instance of the type `IRuleList`.
 *
 * @return An unique pointer to an object of type `IRuleList` that has been created
 */
MLRLCOMMON_API std::unique_ptr<IRuleList> createRuleList();
