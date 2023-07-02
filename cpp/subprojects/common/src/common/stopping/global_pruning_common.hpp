/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/math/math.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/statistics/statistics.hpp"

/**
 * Calculates and returns a numerical score that assesses the quality of a model's predictions for the examples in a
 * training set.
 *
 * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices of the
 *                      examples that are included in the training set
 * @param useHoldoutSet True, if the quality of the predictions should be measured on the holdout set, if available,
 *                      false, if the training set should be used instead
 * @param statistics    A reference to an object of type `IStatistics` that should be used to calculate the quality of
 *                      the predictions
 * @return              The numerical score that has been calculated
 */
static inline float64 evaluate(const SinglePartition& partition, bool useHoldoutSet, const IStatistics& statistics) {
    uint32 numExamples = partition.getNumElements();
    SinglePartition::const_iterator iterator = partition.cbegin();
    float64 mean = 0;

    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = iterator[i];
        float64 score = statistics.evaluatePrediction(exampleIndex);
        mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
    }

    return mean;
}

/**
 * Calculates and returns a numerical score that assesses the quality of a model's predictions for the examples in a
 * training or holdout set, respectively.
 *
 * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of the
 *                      examples that are included in the training and holdout set, respectively
 * @param useHoldoutSet True, if the quality of the predictions should be measured on the holdout set, if available,
 *                      false, if the training set should be used instead
 * @param statistics    A reference to an object of type `IStatistics` that should be used to calculate the quality of
 *                      the predictions
 * @return              The numerical score that has been calculated
 */
static inline float64 evaluate(const BiPartition& partition, bool useHoldoutSet, const IStatistics& statistics) {
    uint32 numExamples;
    BiPartition::const_iterator iterator;

    if (useHoldoutSet) {
        numExamples = partition.getNumSecond();
        iterator = partition.second_cbegin();
    } else {
        numExamples = partition.getNumFirst();
        iterator = partition.first_cbegin();
    }

    float64 mean = 0;

    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = iterator[i];
        float64 score = statistics.evaluatePrediction(exampleIndex);
        mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
    }

    return mean;
}
