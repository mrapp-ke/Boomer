/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include <functional>

// Forward declarations
class CompleteHead;
class PartialHead;


/**
 * Defines an interface for all classes that represent the head of a rule.
 */
class IHead {

    public:

        virtual ~IHead() { };

        /**
         * A visitor function for handling objects of the type `CompleteHead`.
         */
        typedef std::function<void(const CompleteHead&)> CompleteHeadVisitor;

        /**
         * A visitor function for handling objects of the type `PartialHead`.
         */
        typedef std::function<void(const PartialHead&)> PartialHeadVisitor;

        /**
         * Invokes one of the given visitor functions, depending on which one is able to handle this particular type of
         * head.
         *
         * @param completeHeadVisitor   The visitor function for handling objects of the type `CompleteHead`
         * @param partialHeadVisitor    The visitor function for handling objects of the type `PartialHead`
         */
        virtual void visit(CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const = 0;

};
