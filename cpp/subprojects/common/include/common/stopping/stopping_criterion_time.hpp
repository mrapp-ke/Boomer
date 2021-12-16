/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"
#include <chrono>


/**
 * A stopping criterion that ensures that a certain time limit is not exceeded.
 */
class TimeStoppingCriterion final : public IStoppingCriterion {

    private:

        typedef std::chrono::steady_clock timer;

        typedef std::chrono::seconds timer_unit;

        timer_unit timeLimit_;

        std::chrono::time_point<timer> startTime_;

        bool timerStarted_;

    public:

        /**
         * @param timeLimit The time limit in seconds. Must be at least 1
         */
        TimeStoppingCriterion(uint32 timeLimit);

        Result test(const IPartition& partition, const IStatistics& statistics, uint32 numRules) override;

};
