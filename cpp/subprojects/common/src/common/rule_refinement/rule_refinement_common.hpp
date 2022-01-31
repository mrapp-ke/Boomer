/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_evaluation/score_vector.hpp"
#include "common/rule_refinement/prediction_evaluated.hpp"


/**
 * Returns whether the predictions that are stored by a specific `IScoreVector` are better than those of the best head
 * found so far, according to their respective quality scores.
 *
 * @param scoreVector   A reference to an object of type `IScoreVector` that stores the predictions
 * @param bestHead      A pointer to an object of type `AbstractEvaluatedPrediction` that represents the best head found
 *                      so far or a null pointer, if no such head is available
 * @return              True, if the predictions that are stored by the given `IScoreVector` are better than those of
 *                      the best head, false otherwise
 */
static inline constexpr bool isBetterThanBestHead(const IScoreVector& scoreVector,
                                                  const AbstractEvaluatedPrediction* bestHead) {
    return !bestHead || scoreVector.overallQualityScore < bestHead->overallQualityScore;
}
