/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/statistics/statistics.hpp"

/**
 * Defines an interface for all stopping criteria that allow to decide whether additional rules should be induced or
 * not.
 */
class IStoppingCriterion {
    public:

        /**
         * The result that is returned by a stopping criterion. It consists of the action to be executed, as well as the
         * number of rules to be used, if the action is not `CONTINUE`.
         */
        struct Result final {
            public:

                Result() : stop(false), numUsedRules(0) {};

                /**
                 True, if the induction of rules should be stopped, false otherwise.
                 */
                bool stop;

                /**
                 * The number of rules to be used.
                 */
                uint32 numUsedRules;
        };

        virtual ~IStoppingCriterion() {};

        /**
         * Checks whether additional rules should be induced or not.
         *
         * @param statistics    A reference to an object of type `IStatistics` that will serve as the basis for learning
         *                      the next rule
         * @param numRules      The number of rules induced so far
         * @return              A value of the enum `Result` that specifies whether the induction of rules should be
         *                      continued (`CONTINUE`), whether the current number of rules should be stored as a
         *                      potential point for stopping while continuing to induce rules (`STORE_STOP`), or if the
         *                      induction of rules should be forced to be stopped (`FORCE_STOP`)
         */
        virtual Result test(const IStatistics& statistics, uint32 numRules) = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IStoppingCriterion`.
 */
class IStoppingCriterionFactory {
    public:

        virtual ~IStoppingCriterionFactory() {};

        /**
         * Creates and returns a new object of type `IStoppingCriterion`.
         *
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that belong to the training set and the holdout set,
         *                      respectively
         * @return              An unique pointer to an object of type `IStoppingCriterion` that has been created
         */
        virtual std::unique_ptr<IStoppingCriterion> create(const SinglePartition& partition) const = 0;

        /**
         * Creates and returns a new object of type `IStoppingCriterion`.
         *
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that belong to the training set and the holdout set, respectively
         * @return              An unique pointer to an object of type `IStoppingCriterion` that has been created
         */
        virtual std::unique_ptr<IStoppingCriterion> create(BiPartition& partition) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a stopping criterion that allows to decide whether
 * additional rules should be induced or not.
 */
class IStoppingCriterionConfig {
    public:

        virtual ~IStoppingCriterionConfig() {};

        /**
         * Creates and returns a new object of type `IStoppingCriterionFactory` according to the specified
         * configuration.
         *
         * @return An unique pointer to an object of type `IStoppingCriterionFactory` that has been created
         */
        virtual std::unique_ptr<IStoppingCriterionFactory> createStoppingCriterionFactory() const = 0;
};
