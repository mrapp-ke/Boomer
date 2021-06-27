/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/model/body.hpp"
#include "common/model/head.hpp"
#include <memory>


/**
 * A rule that consists of a body and a head.
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
         * Invokes some of the given visitor functions, depending on which ones are able to handle the rule's particular
         * type of body and head.
         *
         * @param emptyBodyVisitor          The visitor function for handling objects of the type `EmptyBody`
         * @param conjunctiveBodyVisitor    The visitor function for handling objects of the type `ConjunctiveBody`
         * @param fullHeadVisitor           The visitor function for handling objects of the type `FullHead`
         * @param partialHeadVisitor        The visitor function for handling objects of the type `PartialHead`
         */
        void visit(IBody::EmptyBodyVisitor emptyBodyVisitor, IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                   IHead::FullHeadVisitor fullHeadVisitor, IHead::PartialHeadVisitor partialHeadVisitor) const;

};
