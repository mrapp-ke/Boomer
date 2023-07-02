/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/data/statistic_vector_example_wise_dense.hpp"
#include "rule_evaluation_label_wise_partial_dynamic_common.hpp"

namespace boosting {

    /**
     * Determines and returns the minimum and maximum absolute score to be predicted for a label. The scores to be
     * predicted for individual labels are also written to a given iterator.
     *
     * @tparam ScoreIterator            The type of the iterator, the scores should be written to
     * @param scoreIterator             An iterator, the scores should be written to
     * @param gradientIterator          An iterator that provides access to the gradient for each label
     * @param hessianIterator           An iterator that provides access to the Hessian for each label
     * @param numLabels                 The total number of available labels
     * @param l1RegularizationWeight    The l2 regularization weight
     * @param l2RegularizationWeight    The L1 regularization weight
     * @return                          A `std::pair` that stores the minimum and maximum absolute score
     */
    template<typename ScoreIterator>
    static inline std::pair<float64, float64> getMinAndMaxScore(
      ScoreIterator scoreIterator, DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator,
      DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessianIterator, uint32 numLabels,
      float64 l1RegularizationWeight, float64 l2RegularizationWeight) {
        float64 score = calculateLabelWiseScore(gradientIterator[0], hessianIterator[0], l1RegularizationWeight,
                                                l2RegularizationWeight);
        scoreIterator[0] = score;
        float64 maxAbsScore = std::abs(score);
        float64 minAbsScore = maxAbsScore;

        for (uint32 i = 1; i < numLabels; i++) {
            score = calculateLabelWiseScore(gradientIterator[i], hessianIterator[i], l1RegularizationWeight,
                                            l2RegularizationWeight);
            scoreIterator[i] = score;
            score = std::abs(score);

            if (score > maxAbsScore) {
                maxAbsScore = score;
            } else if (score < minAbsScore) {
                minAbsScore = score;
            }
        }

        return std::make_pair(minAbsScore, maxAbsScore);
    }
}
