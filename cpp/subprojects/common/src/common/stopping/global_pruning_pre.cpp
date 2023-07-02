#include "common/stopping/global_pruning_pre.hpp"

#include "aggregation_function_common.hpp"
#include "common/util/validation.hpp"
#include "global_pruning_common.hpp"

/**
 * An implementation of the type `IStoppingCriterion` that stops the induction of rules as soon as the quality of a
 * model's predictions for the examples in the training or holdout set do not improve according a certain measure.
 *
 * @tparam Partition The type of the object that provides access to the indices of the examples that are included in the
 *                   training and holdout set, respectively
 */
template<typename Partition>
class PrePruning final : public IStoppingCriterion {
    private:

        const Partition& partition_;

        const std::unique_ptr<IAggregationFunction> aggregationFunctionPtr_;

        const bool useHoldoutSet_;

        const bool removeUnusedRules_;

        const uint32 updateInterval_;

        const uint32 stopInterval_;

        const float64 minImprovement_;

        RingBuffer<float64> pastBuffer_;

        RingBuffer<float64> recentBuffer_;

        uint32 offset_;

        float64 bestScore_;

        uint32 bestNumRules_;

        bool stopped_;

    public:

        /**
         * @param partition                 A reference to an object of template type `Partition` that provides access
         *                                  to the indices of the examples that are included in the training and holdout
         *                                  set, respectively
         * @param aggregationFunctionPtr    An unique pointer to an object of type `IAggregationFunctionFactory` that
         *                                  allows to create implementations of the aggregation function that should be
         *                                  used to aggregate the scores in the buffer
         * @param useHoldoutSet             True, if the quality of the current model's predictions should be measured
         *                                  on the holdout set, if available, false otherwise
         * @param removeUnusedRules         True, if rules that have been induced, but are not used, should be removed
         *                                  from a model, false otherwise
         * @param minRules                  The minimum number of rules that must have been learned until the induction
         *                                  of rules might be stopped. Must be at least 1
         * @param updateInterval            The interval to be used to update the quality of the current model, e.g., a
         *                                  value of 5 means that the model quality is assessed every 5 rules. Must be
         *                                  at least 1
         * @param stopInterval              The interval to be used to decide whether the induction of rules should be
         *                                  stopped, e.g., a value of 10 means that the rule induction might be stopped
         *                                  after 10, 20, ... rules. Must be a multiple of `updateInterval`
         * @param numPast                   The number of past iterations to be stored in a buffer. Must be at least 1
         * @param numCurrent                The number of the most recent iterations to be stored in a buffer. Must be
         *                                  at least 1
         * @param minImprovement            The minimum improvement in percent that must be reached for the rule
         *                                  induction to be continued. Must be in [0, 1]
         */
        PrePruning(const Partition& partition, std::unique_ptr<IAggregationFunction> aggregationFunctionPtr,
                   bool useHoldoutSet, bool removeUnusedRules, uint32 minRules, uint32 updateInterval,
                   uint32 stopInterval, uint32 numPast, uint32 numCurrent, float64 minImprovement)
            : partition_(partition), aggregationFunctionPtr_(std::move(aggregationFunctionPtr)),
              useHoldoutSet_(useHoldoutSet), removeUnusedRules_(removeUnusedRules), updateInterval_(updateInterval),
              stopInterval_(stopInterval), minImprovement_(minImprovement), pastBuffer_(RingBuffer<float64>(numPast)),
              recentBuffer_(RingBuffer<float64>(numCurrent)), bestScore_(std::numeric_limits<float64>::infinity()),
              stopped_(false) {
            uint32 bufferInterval = (numPast * updateInterval) + (numCurrent * updateInterval);
            offset_ = bufferInterval < minRules ? minRules - bufferInterval : 0;
        }

        Result test(const IStatistics& statistics, uint32 numRules) override {
            Result result;

            if (!stopped_ && numRules > offset_ && numRules % updateInterval_ == 0) {
                float64 currentScore = evaluate(partition_, useHoldoutSet_, statistics);

                if (pastBuffer_.isFull()) {
                    if (currentScore < bestScore_) {
                        bestScore_ = currentScore;
                        bestNumRules_ = numRules;
                    }

                    if (numRules % stopInterval_ == 0) {
                        float64 aggregatedScorePast =
                          aggregationFunctionPtr_->aggregate(pastBuffer_.cbegin(), pastBuffer_.cend());
                        float64 aggregatedScoreRecent =
                          aggregationFunctionPtr_->aggregate(recentBuffer_.cbegin(), recentBuffer_.cend());
                        float64 percentageImprovement =
                          (aggregatedScorePast - aggregatedScoreRecent) / aggregatedScoreRecent;

                        if (percentageImprovement <= minImprovement_) {
                            result.stop = removeUnusedRules_;
                            result.numUsedRules = bestNumRules_;
                            stopped_ = true;
                        }
                    }
                }

                std::pair<bool, float64> pair = recentBuffer_.push(currentScore);

                if (pair.first) {
                    pastBuffer_.push(pair.second);
                }
            }

            return result;
        }
};

/**
 * Allows to create implementations of the type `IStoppingCriterion` that stop the induction of rules as soon as the
 * quality of a model's predictions for the examples in the training or holdout set do not improve according a certain
 * measure.
 */
class PrePruningFactory final : public IStoppingCriterionFactory {
    private:

        std::unique_ptr<IAggregationFunctionFactory> aggregationFunctionFactoryPtr_;

        bool useHoldoutSet_;

        bool removeUnusedRules_;

        uint32 minRules_;

        uint32 updateInterval_;

        uint32 stopInterval_;

        uint32 numPast_;

        uint32 numCurrent_;

        float64 minImprovement_;

    public:

        /**
         * @param aggregationFunctionFactoryPtr An unique pointer to an object of type `IAggregationFunctionFactory`
         *                                      that allows to create implementations of the aggregation function that
         *                                      should be used to aggregate the scores in the buffer
         * @param useHoldoutSet                 True, if the quality of the current model's predictions should be
         *                                      measured on the holdout set, if available, false otherwise
         * @param removeUnusedRules             True, if rules that have been induced, but are not used, should be
         *                                      removed from a model, false otherwise
         * @param minRules                      The minimum number of rules that must have been learned until the
         *                                      induction of rules might be stopped. Must be at least 1
         * @param updateInterval                The interval to be used to update the quality of the current model,
         *                                      e.g., a value of 5 means that the model quality is assessed every 5
         *                                      rules. Must be at least 1
         * @param stopInterval                  The interval to be used to decide whether the induction of rules should
         *                                      be stopped, e.g., a value of 10 means that the rule induction might be
         *                                      stopped after 10, 20, ... rules. Must be a multiple of `updateInterval`
         * @param numPast                       The number of past iterations to be stored in a buffer. Must be at least
         *                                      1
         * @param numCurrent                    The number of the most recent iterations to be stored in a buffer. Must
         *                                      be at least 1
         * @param minImprovement                The minimum improvement in percent that must be reached for the rule
         *                                      induction to be continued. Must be in [0, 1]
         */
        PrePruningFactory(std::unique_ptr<IAggregationFunctionFactory> aggregationFunctionFactoryPtr,
                          bool useHoldoutSet, bool removeUnusedRules, uint32 minRules, uint32 updateInterval,
                          uint32 stopInterval, uint32 numPast, uint32 numCurrent, float64 minImprovement)
            : aggregationFunctionFactoryPtr_(std::move(aggregationFunctionFactoryPtr)), useHoldoutSet_(useHoldoutSet),
              minRules_(minRules), updateInterval_(updateInterval), stopInterval_(stopInterval), numPast_(numPast),
              numCurrent_(numCurrent), minImprovement_(minImprovement) {}

        std::unique_ptr<IStoppingCriterion> create(const SinglePartition& partition) const override {
            std::unique_ptr<IAggregationFunction> aggregationFunctionPtr = aggregationFunctionFactoryPtr_->create();
            return std::make_unique<PrePruning<const SinglePartition>>(
              partition, std::move(aggregationFunctionPtr), useHoldoutSet_, removeUnusedRules_, minRules_,
              updateInterval_, stopInterval_, numPast_, numCurrent_, minImprovement_);
        }

        std::unique_ptr<IStoppingCriterion> create(BiPartition& partition) const override {
            std::unique_ptr<IAggregationFunction> aggregationFunctionPtr = aggregationFunctionFactoryPtr_->create();
            return std::make_unique<PrePruning<BiPartition>>(
              partition, std::move(aggregationFunctionPtr), useHoldoutSet_, removeUnusedRules_, minRules_,
              updateInterval_, stopInterval_, numPast_, numCurrent_, minImprovement_);
        }
};

PrePruningConfig::PrePruningConfig()
    : aggregationFunction_(AggregationFunction::ARITHMETIC_MEAN), useHoldoutSet_(true), removeUnusedRules_(true),
      minRules_(100), updateInterval_(1), stopInterval_(1), numPast_(50), numCurrent_(50), minImprovement_(0.005) {}

AggregationFunction PrePruningConfig::getAggregationFunction() const {
    return aggregationFunction_;
}

IPrePruningConfig& PrePruningConfig::setAggregationFunction(AggregationFunction aggregationFunction) {
    aggregationFunction_ = aggregationFunction;
    return *this;
}

bool PrePruningConfig::isHoldoutSetUsed() const {
    return useHoldoutSet_;
}

IPrePruningConfig& PrePruningConfig::setUseHoldoutSet(bool useHoldoutSet) {
    useHoldoutSet_ = useHoldoutSet;
    return *this;
}

bool PrePruningConfig::isRemoveUnusedRules() const {
    return removeUnusedRules_;
}

IPrePruningConfig& PrePruningConfig::setRemoveUnusedRules(bool removeUnusedRules) {
    removeUnusedRules_ = removeUnusedRules;
    return *this;
}

uint32 PrePruningConfig::getMinRules() const {
    return minRules_;
}

IPrePruningConfig& PrePruningConfig::setMinRules(uint32 minRules) {
    assertGreaterOrEqual<uint32>("minRules", minRules, 1);
    minRules_ = minRules;
    return *this;
}

uint32 PrePruningConfig::getUpdateInterval() const {
    return updateInterval_;
}

IPrePruningConfig& PrePruningConfig::setUpdateInterval(uint32 updateInterval) {
    assertGreaterOrEqual<uint32>("updateInterval", updateInterval, 1);
    updateInterval_ = updateInterval;
    return *this;
}

uint32 PrePruningConfig::getStopInterval() const {
    return stopInterval_;
}

IPrePruningConfig& PrePruningConfig::setStopInterval(uint32 stopInterval) {
    assertMultiple<uint32>("stopInterval", stopInterval, updateInterval_);
    stopInterval_ = stopInterval;
    return *this;
}

uint32 PrePruningConfig::getNumPast() const {
    return numPast_;
}

IPrePruningConfig& PrePruningConfig::setNumPast(uint32 numPast) {
    assertGreaterOrEqual<uint32>("numPast", numPast, 1);
    numPast_ = numPast;
    return *this;
}

uint32 PrePruningConfig::getNumCurrent() const {
    return numCurrent_;
}

IPrePruningConfig& PrePruningConfig::setNumCurrent(uint32 numCurrent) {
    assertGreaterOrEqual<uint32>("numCurrent", numCurrent, 1);
    numCurrent_ = numCurrent;
    return *this;
}

float64 PrePruningConfig::getMinImprovement() const {
    return minImprovement_;
}

IPrePruningConfig& PrePruningConfig::setMinImprovement(float64 minImprovement) {
    assertGreaterOrEqual<float64>("minImprovement", minImprovement, 0);
    assertLessOrEqual<float64>("minImprovement", minImprovement, 1);
    minImprovement_ = minImprovement;
    return *this;
}

std::unique_ptr<IStoppingCriterionFactory> PrePruningConfig::createStoppingCriterionFactory() const {
    std::unique_ptr<IAggregationFunctionFactory> aggregationFunctionFactoryPtr =
      createAggregationFunctionFactory(aggregationFunction_);
    return std::make_unique<PrePruningFactory>(std::move(aggregationFunctionFactoryPtr), useHoldoutSet_,
                                               removeUnusedRules_, minRules_, updateInterval_, stopInterval_, numPast_,
                                               numCurrent_, minImprovement_);
}

bool PrePruningConfig::shouldUseHoldoutSet() const {
    return useHoldoutSet_;
}

bool PrePruningConfig::shouldRemoveUnusedRules() const {
    return removeUnusedRules_;
}
