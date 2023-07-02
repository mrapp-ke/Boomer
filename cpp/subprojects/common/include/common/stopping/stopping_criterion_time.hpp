/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/macros.hpp"
#include "common/stopping/stopping_criterion.hpp"

/**
 * Defines an interface for all classes that allow to configure a stopping criterion that ensures that a certain time
 * limit is not exceeded.
 */
class MLRLCOMMON_API ITimeStoppingCriterionConfig {
    public:

        virtual ~ITimeStoppingCriterionConfig() {};

        /**
         * Returns the time limit.
         *
         * @return The time limit in seconds
         */
        virtual uint32 getTimeLimit() const = 0;

        /**
         * Sets the time limit.
         *
         * @param timeLimit The time limit in seconds. Must be at least 1
         * @return          A reference to an object of type `ITimeStoppingCriterionConfig` that allows further
         *                  configuration of the stopping criterion
         */
        virtual ITimeStoppingCriterionConfig& setTimeLimit(uint32 timeLimit) = 0;
};

/**
 * Allows to configure a stopping criterion that ensures that a certain time limit is not exceeded.
 */
class TimeStoppingCriterionConfig final : public IStoppingCriterionConfig,
                                          public ITimeStoppingCriterionConfig {
    private:

        uint32 timeLimit_;

    public:

        TimeStoppingCriterionConfig();

        uint32 getTimeLimit() const override;

        ITimeStoppingCriterionConfig& setTimeLimit(uint32 timeLimit) override;

        std::unique_ptr<IStoppingCriterionFactory> createStoppingCriterionFactory() const override;
};
