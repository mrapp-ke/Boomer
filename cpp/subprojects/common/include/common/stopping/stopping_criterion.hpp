/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/partition.hpp"
#include "common/statistics/statistics.hpp"


/**
 * Defines an interface for all stopping criteria that allow to decide whether additional rules should be induced or
 * not.
 */
class IStoppingCriterion {

    public:

        /**
         * An enum that specifies all possible actions that may be executed, based on the result that is returned by a
         * stopping criterion.
         */
        enum Action : uint32 {
            CONTINUE = 0,
            STORE_STOP = 1,
            FORCE_STOP = 2
        };

        /**
         * The result that is returned by a stopping criterion. It consists of the action to be executed, as well as the
         * number of rules to be used, if the action is not `CONTINUE`.
         */
        struct Result {

            /**
             * The action to be executed.
             */
            Action action;

            /**
             * The number of rules to be used.
             */
            uint32 numRules;

        };

        virtual ~IStoppingCriterion() { };

        /**
         * Checks whether additional rules should be induced or not.
         *
         * @param partition     A reference to an object of type `IPartition` that provides access to the indices of the
         *                      training examples that belong to the training set and the holdout set, respectively
         * @param statistics    A reference to an object of type `IStatistics` that will serve as the basis for learning
         *                      the next rule
         * @param numRules      The number of rules induced so far
         * @return              A value of the enum `Result` that specifies whether the induction of rules should be
         *                      continued (`CONTINUE`), whether the current number of rules should be stored as a
         *                      potential point for stopping while continuing to induce rules (`STORE_STOP`), or if the
         *                      induction of rules should be forced to be stopped (`FORCE_STOP`)
         */
        virtual Result test(const IPartition& partition, const IStatistics& statistics, uint32 numRules) = 0;

};
