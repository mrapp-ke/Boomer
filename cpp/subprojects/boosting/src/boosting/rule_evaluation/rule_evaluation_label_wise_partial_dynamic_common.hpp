/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/data/statistic_vector_label_wise_dense.hpp"
#include "rule_evaluation_label_wise_common.hpp"

#include <utility>

namespace boosting {

    /**
     * Determines and returns the minimum and maximum absolute score to be predicted for a label.
     *
     * @tparam StatisticIterator        The type of the iterator that provides access to the gradients and Hessians
     * @param statisticIterator         An iterator that provides access to the gradients and Hessians for each label
     * @param numLabels                 The total number of available labels
     * @param l1RegularizationWeight    The l2 regularization weight
     * @param l2RegularizationWeight    The L1 regularization weight
     * @return                          A `std::pair` that stores the minimum and maximum absolute score
     */
    template<typename StatisticIterator>
    static inline std::pair<float64, float64> getMinAndMaxScore(StatisticIterator& statisticIterator, uint32 numLabels,
                                                                float64 l1RegularizationWeight,
                                                                float64 l2RegularizationWeight) {
        const Tuple<float64>& firstTuple = statisticIterator[0];
        float64 maxAbsScore = std::abs(
          calculateLabelWiseScore(firstTuple.first, firstTuple.second, l1RegularizationWeight, l2RegularizationWeight));
        float64 minAbsScore = maxAbsScore;

        for (uint32 i = 1; i < numLabels; i++) {
            const Tuple<float64>& tuple = statisticIterator[i];
            float64 absScore = std::abs(
              calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight, l2RegularizationWeight));

            if (absScore > maxAbsScore) {
                maxAbsScore = absScore;
            } else if (absScore < minAbsScore) {
                minAbsScore = absScore;
            }
        }

        return std::make_pair(minAbsScore, maxAbsScore);
    }

    /**
     * Calculates and returns the threshold that should be used to decide whether a rule should predict for a label or
     * not.
     *
     * @param minAbsScore   The minimum absolute score to be predicted for a label
     * @param maxAbsScore   The maximum absolute score to be predicted for a label
     * @param threshold     A threshold that affects for how many labels the rule heads should predict
     * @param exponent      An exponent that is used to weigh the estimated predictive quality for individual labels
     * @return              The threshold that has been calculated
     */
    static inline float64 calculateThreshold(float64 minAbsScore, float64 maxAbsScore, float64 threshold,
                                             float64 exponent) {
        return std::pow(maxAbsScore - minAbsScore, exponent) * threshold;
    }

    /**
     * Weighs and returns the score that is predicted for a particular label, depending on the minimum absolute score
     * that has been determined via the function `getMinMaxScore` and a given exponent.
     *
     * @param score         The score to be predicted
     * @param minAbsScore   The minimum absolute score to be predicted for a label
     * @param exponent      An exponent that is used to weigh the estimated predictive quality for individual labels
     * @return              The weighted score that has been calculated
     */
    static inline float64 calculateWeightedScore(float64 score, float64 minAbsScore, float64 exponent) {
        return std::pow(std::abs(score) - minAbsScore, exponent);
    }

}
