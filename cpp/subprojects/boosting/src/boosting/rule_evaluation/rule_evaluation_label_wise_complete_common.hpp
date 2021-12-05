/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/data/statistic_vector_label_wise_dense.hpp"
#include "rule_evaluation_label_wise_common.hpp"


namespace boosting {

    /**
     * Calculates the optimal scores to be predicted for several labels, based on the corresponding gradients and
     * Hessians and taking L1 and L2 regularization into account, and writes them to an iterator.
     *
     * @tparam ScoreIterator            The type of the iterator, the calculated scores should be written to
     * @param statisticIterator         A `DenseLabelWiseStatisticVector::const_iterator` that provides access to the
     *                                  gradients and Hessians
     * @param scoreIterator             An iterator that allows to write the calculated scores
     * @param numElements               The number of scores to be calculated
     * @param l1RegularizationWeight    The weight of the L1 regularization
     * @param l2RegularizationWeight    The weight of the L2 regularization
     */
    template<typename ScoreIterator>
    static inline void calculateLabelWiseScores(DenseLabelWiseStatisticVector::const_iterator statisticIterator,
                                                ScoreIterator scoreIterator, uint32 numElements,
                                                float64 l1RegularizationWeight, float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < numElements; i++) {
            const Tuple<float64>& tuple = statisticIterator[i];
            scoreIterator[i] = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight,
                                                       l2RegularizationWeight);
        }
    }

    /**
     * Calculates and returns a quality score that assesses the quality of the score that is predicted for a single
     * label, taking L1 and L2 regularization into account.
     *
     * @param score                     The predicted score
     * @param gradient                  The gradient
     * @param hessian                   The Hessian
     * @param l1RegularizationWeight    The weight of the L1 regularization
     * @param l2RegularizationWeight    The weight of the L2 regularization
     * @return                          The quality score that has been calculated
     */
    static inline constexpr float64 calculateLabelWiseQualityScore(float64 score, float64 gradient, float64 hessian,
                                                                   float64 l1RegularizationWeight,
                                                                   float64 l2RegularizationWeight) {
        float64 scorePow = score * score;
        float64 qualityScore =  (gradient * score) + (0.5 * hessian * scorePow);
        float64 l1RegularizationTerm = l1RegularizationWeight * std::abs(score);
        float64 l2RegularizationTerm = 0.5 * l2RegularizationWeight * scorePow;
        return qualityScore + l1RegularizationTerm + l2RegularizationTerm;
    }

}
