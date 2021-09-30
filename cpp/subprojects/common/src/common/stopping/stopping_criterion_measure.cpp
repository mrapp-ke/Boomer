#include "common/stopping/stopping_criterion_measure.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/math/math.hpp"
#include "common/validation.hpp"
#include <limits>


static inline float64 evaluateOnHoldoutSet(const BiPartition& partition, const IStatistics& statistics) {
    uint32 numHoldoutExamples = partition.getNumSecond();
    BiPartition::const_iterator iterator = partition.second_cbegin();
    float64 mean = 0;

    for (uint32 i = 0; i < numHoldoutExamples; i++) {
        uint32 exampleIndex = iterator[i];
        float64 score = statistics.evaluatePrediction(exampleIndex);
        mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
    }

    return mean;
}

float64 MinFunction::aggregate(RingBuffer<float64>::const_iterator begin,
                               RingBuffer<float64>::const_iterator end) const {
    uint32 numElements = end - begin;
    float64 min = begin[0];

    for (uint32 i = 1; i < numElements; i++) {
        float64 value = begin[i];

        if (value < min) {
            min = value;
        }
    }

    return min;
}

float64 MaxFunction::aggregate(RingBuffer<float64>::const_iterator begin,
                               RingBuffer<float64>::const_iterator end) const {
    uint32 numElements = end - begin;
    float64 max = begin[0];

    for (uint32 i = 1; i < numElements; i++) {
        float64 value = begin[i];

        if (value > max) {
            max = value;
        }
    }

    return max;
}

float64 ArithmeticMeanFunction::aggregate(RingBuffer<float64>::const_iterator begin,
                                          RingBuffer<float64>::const_iterator end) const {
    uint32 numElements = end - begin;
    float64 mean = 0;

    for (uint32 i = 0; i < numElements; i++) {
        float64 value = begin[i];
        mean = iterativeArithmeticMean<float64>(i + 1, value, mean);
    }

    return mean;
}

MeasureStoppingCriterion::MeasureStoppingCriterion(std::unique_ptr<IAggregationFunction> aggregationFunctionPtr,
                                                   uint32 minRules, uint32 updateInterval, uint32 stopInterval,
                                                   uint32 numPast, uint32 numCurrent, float64 minImprovement,
                                                   bool forceStop)
    : aggregationFunctionPtr_(std::move(aggregationFunctionPtr)), updateInterval_(updateInterval),
      stopInterval_(stopInterval), minImprovement_(minImprovement), pastBuffer_(RingBuffer<float64>(numPast)),
      recentBuffer_(RingBuffer<float64>(numCurrent)), stoppingAction_(forceStop ? FORCE_STOP : STORE_STOP),
      bestScore_(std::numeric_limits<float64>::infinity()), stopped_(false) {
    assertNotNull("aggregationFunctionPtr", aggregationFunctionPtr_.get());
    assertGreaterOrEqual<uint32>("minRules", minRules, 1);
    assertGreaterOrEqual<uint32>("updateInterval", updateInterval, 1);
    assertMultiple<uint32>("stopInterval", stopInterval, updateInterval);
    assertGreaterOrEqual<uint32>("numPast", numPast, 1);
    assertGreaterOrEqual<uint32>("numCurrent", numCurrent, 1);
    assertGreaterOrEqual<float64>("minImprovement", minImprovement, 0);
    assertLessOrEqual<float64>("minImprovement", minImprovement, 1);
    uint32 bufferInterval = (numPast * updateInterval) + (numCurrent * updateInterval);
    offset_ = bufferInterval < minRules ? minRules - bufferInterval : 0;
}

IStoppingCriterion::Result MeasureStoppingCriterion::test(const IPartition& partition, const IStatistics& statistics,
                                                          uint32 numRules) {
    Result result;
    result.action = CONTINUE;

    if (!stopped_ && numRules > offset_ && numRules % updateInterval_ == 0) {
        const BiPartition& biPartition = static_cast<const BiPartition&>(partition);
        float64 currentScore = evaluateOnHoldoutSet(biPartition, statistics);

        if (pastBuffer_.isFull()) {
            if (currentScore < bestScore_) {
                bestScore_ = currentScore;
                bestNumRules_ = numRules;
            }

            if (numRules % stopInterval_ == 0) {
                float64 aggregatedScorePast = aggregationFunctionPtr_->aggregate(pastBuffer_.cbegin(),
                                                                                 pastBuffer_.cend());
                float64 aggregatedScoreRecent = aggregationFunctionPtr_->aggregate(recentBuffer_.cbegin(),
                                                                                   recentBuffer_.cend());
                float64 percentageImprovement = (aggregatedScorePast - aggregatedScoreRecent) / aggregatedScoreRecent;

                if (percentageImprovement <= minImprovement_) {
                    result.action = stoppingAction_;
                    result.numRules = bestNumRules_;
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
