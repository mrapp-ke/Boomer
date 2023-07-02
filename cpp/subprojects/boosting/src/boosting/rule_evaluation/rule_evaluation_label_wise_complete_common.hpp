/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_evaluation/score_vector_dense.hpp"
#include "rule_evaluation_label_wise_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of complete rules, as well as their overall quality, based on the gradients
     * and Hessians that are stored by a vector using L1 and L2 regularization.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the labels for which predictions should
     *                          be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class LabelWiseCompleteRuleEvaluation final : public IRuleEvaluation<StatisticVector> {
        private:

            DenseScoreVector<IndexVector> scoreVector_;

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the labels for which the rules may predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            LabelWiseCompleteRuleEvaluation(const IndexVector& labelIndices, float64 l1RegularizationWeight,
                                            float64 l2RegularizationWeight)
                : scoreVector_(DenseScoreVector<IndexVector>(labelIndices, true)),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

            const IScoreVector& calculateScores(StatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                typename StatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                typename DenseScoreVector<IndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                float64 quality = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    float64 predictedScore = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight_,
                                                                     l2RegularizationWeight_);
                    scoreIterator[i] = predictedScore;
                    quality += calculateLabelWiseQuality(predictedScore, tuple.first, tuple.second,
                                                         l1RegularizationWeight_, l2RegularizationWeight_);
                }

                scoreVector_.quality = quality;
                return scoreVector_;
            }
    };

}