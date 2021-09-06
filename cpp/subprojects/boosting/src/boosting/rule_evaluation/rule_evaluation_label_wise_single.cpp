#include "boosting/rule_evaluation/rule_evaluation_label_wise_single.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"
#include "common/validation.hpp"
#include "rule_evaluation_label_wise_common.hpp"


namespace boosting {

    /**
     * Allows to calculate the predictions of single-label rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `DenseLabelWiseStatisticVector` using L2 regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class LabelWiseSingleLabelRuleEvaluation final : public IRuleEvaluation<DenseLabelWiseStatisticVector> {

        private:

            const T& labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            LabelWiseSingleLabelRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight)
                : labelIndices_(labelIndices), indexVector_(PartialIndexVector(1)),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_)),
                  l2RegularizationWeight_(l2RegularizationWeight) {

            }

            const IScoreVector& calculatePrediction(DenseLabelWiseStatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                DenseLabelWiseStatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                const Tuple<float64>& firstTuple = statisticIterator[0];
                float64 bestScore = calculateLabelWiseScore(firstTuple.first, firstTuple.second,
                                                            l2RegularizationWeight_);
                float64 bestAbsScore = std::abs(bestScore);
                uint32 bestIndex = 0;

                for (uint32 i = 1; i < numElements; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    float64 score = calculateLabelWiseScore(tuple.first, tuple.second, l2RegularizationWeight_);
                    float64 absScore = std::abs(score);

                    if (absScore > bestAbsScore) {
                        bestIndex = i;
                        bestScore = score;
                        bestAbsScore = absScore;
                    }
                }

                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                scoreIterator[0] = bestScore;
                indexVector_.begin()[0] = labelIndices_.cbegin()[bestIndex];
                scoreVector_.overallQualityScore = calculateLabelWiseQualityScore(bestScore,
                                                                                  statisticIterator[bestIndex].first,
                                                                                  statisticIterator[bestIndex].second,
                                                                                  l2RegularizationWeight_);
                return scoreVector_;
            }

    };

    LabelWiseSingleLabelRuleEvaluationFactory::LabelWiseSingleLabelRuleEvaluationFactory(float64 l2RegularizationWeight)
        : l2RegularizationWeight_(l2RegularizationWeight) {
        assertGreaterOrEqual<float64>("l2RegularizationWeight", l2RegularizationWeight, 0);
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseSingleLabelRuleEvaluationFactory::createDense(
            const CompleteIndexVector& indexVector) const {
        return std::make_unique<LabelWiseSingleLabelRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                         l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseSingleLabelRuleEvaluationFactory::createDense(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWiseSingleLabelRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                        l2RegularizationWeight_);
    }

}