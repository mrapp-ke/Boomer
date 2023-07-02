#include "common/stopping/stopping_criterion_time.hpp"

#include "common/util/validation.hpp"

#include <chrono>

/**
 * An implementation of the type `IStoppingCriterion` that ensures that a certain time limit is not exceeded.
 */
class TimeStoppingCriterion final : public IStoppingCriterion {
    private:

        typedef std::chrono::steady_clock timer;

        typedef std::chrono::seconds timer_unit;

        const timer_unit timeLimit_;

        std::chrono::time_point<timer> startTime_;

        bool timerStarted_;

    public:

        /**
         * @param timeLimit The time limit in seconds. Must be at least 1
         */
        TimeStoppingCriterion(uint32 timeLimit)
            : timeLimit_(std::chrono::duration_cast<timer_unit>(std::chrono::seconds(timeLimit))),
              startTime_(timer::now()), timerStarted_(false) {}

        Result test(const IStatistics& statistics, uint32 numRules) override {
            Result result;

            if (timerStarted_) {
                auto currentTime = timer::now();
                auto duration = std::chrono::duration_cast<timer_unit>(currentTime - startTime_);

                if (duration >= timeLimit_) {
                    result.stop = true;
                }
            } else {
                startTime_ = timer::now();
                timerStarted_ = true;
            }

            return result;
        }
};

/**
 * Allows to create instances of the type `IStoppingCriterion` that ensure that a certain time limit is not exceeded.
 */
class TimeStoppingCriterionFactory final : public IStoppingCriterionFactory {
    private:

        const uint32 timeLimit_;

    public:

        /**
         * @param timeLimit The time limit in seconds. Must be at least 1
         */
        TimeStoppingCriterionFactory(uint32 timeLimit) : timeLimit_(timeLimit) {}

        std::unique_ptr<IStoppingCriterion> create(const SinglePartition& partition) const override {
            return std::make_unique<TimeStoppingCriterion>(timeLimit_);
        }

        std::unique_ptr<IStoppingCriterion> create(BiPartition& partition) const override {
            return std::make_unique<TimeStoppingCriterion>(timeLimit_);
        }
};

TimeStoppingCriterionConfig::TimeStoppingCriterionConfig() : timeLimit_(3600) {}

uint32 TimeStoppingCriterionConfig::getTimeLimit() const {
    return timeLimit_;
}

ITimeStoppingCriterionConfig& TimeStoppingCriterionConfig::setTimeLimit(uint32 timeLimit) {
    assertGreaterOrEqual<uint32>("timeLimit", timeLimit, 1);
    timeLimit_ = timeLimit;
    return *this;
}

std::unique_ptr<IStoppingCriterionFactory> TimeStoppingCriterionConfig::createStoppingCriterionFactory() const {
    return std::make_unique<TimeStoppingCriterionFactory>(timeLimit_);
}
