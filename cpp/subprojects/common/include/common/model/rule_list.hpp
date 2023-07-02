/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/model/body.hpp"
#include "common/model/head.hpp"
#include "common/model/rule_model.hpp"

#include <vector>

/**
 * Defines an interface for all rule-based models that store several rules in an ordered list. Optionally, the model may
 * also contain a default rule that either takes precedence over the remaining rules or not.
 */
class MLRLCOMMON_API IRuleList : public IRuleModel {
    public:

        virtual ~IRuleList() override {};

        /**
         * Creates a new default rule from a given head and adds it to the model.
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
         * Returns whether the default rule takes precedence over the remaining rules or not.
         *
         * @return True, if the default rule takes precedence over the remaining rules, false otherwise
         */
        virtual bool isDefaultRuleTakingPrecedence() const = 0;

        /**
         * Invokes some of the given visitor functions, depending on which ones are able to handle the bodies and heads
         * of all rules that are contained in this model, including the default rule, if available.
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
         * of all used rules that are contained in this model, including the default rule, if available.
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
 * An implementation of the type `IRuleList` that stores several rules in the order of their induction. Optionally, the
 * model may also contain a default rule that either takes precedence over the remaining rules or not.
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
         * A forward iterator that provides access to the rules in a model, including the default rule, if available.
         */
        class ConstIterator final {
            private:

                const Rule* defaultRule_;

                std::vector<Rule>::const_iterator iterator_;

                uint32 offset_;

                uint32 defaultRuleIndex_;

                uint32 index_;

            public:

                /**
                 * @param defaultRuleTakesPrecedence    True, if the default rule takes precedence over the remaining
                 *                                      rules, false otherwise
                 * @param defaultRule                   A pointer to an object of type `Rule` that stores the default
                 *                                      rule or a null pointer, if no default rule is available
                 * @param iterator                      An iterator to the beginning of the remaining rules
                 * @param start                         The index of the rule to start at
                 * @param end                           The index of the rule to end at (exclusive)
                 */
                ConstIterator(bool defaultRuleTakesPrecedence, const Rule* defaultRule,
                              const std::vector<Rule>::const_iterator iterator, uint32 start, uint32 end);

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
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator that refers to the next element
                 */
                ConstIterator& operator++();

                /**
                 * Returns an iterator to the next element.
                 *
                 * @return A reference to an iterator that refers to the next element
                 */
                ConstIterator& operator++(int n);

                /**
                 * Returns an iterator to one of the subsequent elements.
                 *
                 * @param difference    The number of elements to increment the iterator by
                 * @return              A copy of this iterator that refers to the specified element
                 */
                ConstIterator operator+(const uint32 difference) const;

                /**
                 * Returns an iterator to one of the subsequent elements.
                 *
                 * @param difference    The number of elements to increment the iterator by
                 * @return              A reference to an iterator that refers to the specified element
                 */
                ConstIterator& operator+=(const uint32 difference);

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators do not refer to the same element, false otherwise
                 */
                bool operator!=(const ConstIterator& rhs) const;

                /**
                 * Returns whether this iterator and another one refer to the same element.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      True, if the iterators refer to the same element, false otherwise
                 */
                bool operator==(const ConstIterator& rhs) const;

                /**
                 * Returns the difference between this iterator and another one.
                 *
                 * @param rhs   A reference to another iterator
                 * @return      The difference between the iterators
                 */
                difference_type operator-(const ConstIterator& rhs) const;
        };

        std::unique_ptr<Rule> defaultRulePtr_;

        std::vector<Rule> ruleList_;

        uint32 numUsedRules_;

        bool defaultRuleTakesPrecedence_;

    public:

        /**
         * @param defaultRuleTakesPrecedence True, if the default rule should take precedence over the remaining rules,
         *                                   false otherwise
         */
        RuleList(bool defaultRuleTakesPrecedence);

        /**
         * An iterator that provides read-only access to rules.
         */
        typedef ConstIterator const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of all rules, including the default rule, if available.
         *
         * @param maxRules  The maximum number of rules to consider or 0, if all rules should be considered
         * @return          A `const_iterator` to the beginning
         */
        const_iterator cbegin(uint32 maxRules = 0) const;

        /**
         * Returns a `const_iterator` to the end of all rules, including the default rule, if available.
         *
         * @param maxRules  The maximum number of rules to consider or 0, if all rules should be considered
         * @return          A `const_iterator` to the end
         */
        const_iterator cend(uint32 maxRules = 0) const;

        /**
         * Returns a `const_iterator` to the beginning of all used rules, including the default rule, if available.
         *
         * @param maxRules  The maximum number of rules to consider or 0, if all rules should be considered
         * @return          A `const_iterator` to the beginning
         */
        const_iterator used_cbegin(uint32 maxRules = 0) const;

        /**
         * Returns a `const_iterator` to the end of all used rules, including the default rule, if available.
         *
         * @param maxRules  The maximum number of rules to consider or 0, if all used rules should be considered
         * @return          A `const_iterator` to the end
         */
        const_iterator used_cend(uint32 maxRules = 0) const;

        uint32 getNumRules() const override;

        uint32 getNumUsedRules() const override;

        void setNumUsedRules(uint32 numUsedRules) override;

        void addDefaultRule(std::unique_ptr<IHead> headPtr) override;

        void addRule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr) override;

        bool containsDefaultRule() const override;

        bool isDefaultRuleTakingPrecedence() const override;

        void visit(IBody::EmptyBodyVisitor emptyBodyVisitor, IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                   IHead::CompleteHeadVisitor completeHeadVisitor,
                   IHead::PartialHeadVisitor partialHeadVisitor) const override;

        void visitUsed(IBody::EmptyBodyVisitor emptyBodyVisitor, IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                       IHead::CompleteHeadVisitor completeHeadVisitor,
                       IHead::PartialHeadVisitor partialHeadVisitor) const override;

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CContiguousFeatureMatrix& featureMatrix,
                                                              const ILabelSpaceInfo& labelSpaceInfo,
                                                              uint32 numLabels) const override;

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CsrFeatureMatrix& featureMatrix,
                                                              const ILabelSpaceInfo& labelSpaceInfo,
                                                              uint32 numLabels) const override;

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;
};

/**
 * Creates and returns a new instance of the type `IRuleList`.
 *
 * @param defaultRuleTakesPrecedence    True, if the default rule should take precedence over the remaining rules, false
 *                                      otherwise
 * @return                              An unique pointer to an object of type `IRuleList` that has been created
 */
MLRLCOMMON_API std::unique_ptr<IRuleList> createRuleList(bool defaultRuleTakesPrecedence);
