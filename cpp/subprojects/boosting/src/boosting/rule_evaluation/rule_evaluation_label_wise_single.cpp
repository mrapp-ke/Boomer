#include "boosting/rule_evaluation/rule_evaluation_label_wise_single.hpp"

#include "common/rule_evaluation/score_vector_dense.hpp"
#include "rule_evaluation_label_wise_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of single-label rules, as well as their overall quality, based on the
     * gradients and Hessians that are stored by a vector using L1 and L2 regularization.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the labels for which predictions should
     *                          be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class LabelWiseSingleLabelRuleEvaluation final : public IRuleEvaluation<StatisticVector> {
        private:

            const IndexVector& labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

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
            LabelWiseSingleLabelRuleEvaluation(const IndexVector& labelIndices, float64 l1RegularizationWeight,
                                               float64 l2RegularizationWeight)
                : labelIndices_(labelIndices), indexVector_(PartialIndexVector(1)),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_, true)),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

            const IScoreVector& calculateScores(StatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                typename StatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                const Tuple<float64>& firstTuple = statisticIterator[0];
                float64 bestScore = calculateLabelWiseScore(firstTuple.first, firstTuple.second,
                                                            l1RegularizationWeight_, l2RegularizationWeight_);
                uint32 bestIndex = 0;

                for (uint32 i = 1; i < numElements; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    float64 score = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight_,
                                                            l2RegularizationWeight_);

                    if (std::abs(score) > std::abs(bestScore)) {
                        bestIndex = i;
                        bestScore = score;
                    }
                }

                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                scoreIterator[0] = bestScore;
                indexVector_.begin()[0] = labelIndices_.cbegin()[bestIndex];
                scoreVector_.quality = calculateLabelWiseQuality(bestScore, statisticIterator[bestIndex].first,
                                                                 statisticIterator[bestIndex].second,
                                                                 l1RegularizationWeight_, l2RegularizationWeight_);
                return scoreVector_;
            }
    };

    LabelWiseSingleLabelRuleEvaluationFactory::LabelWiseSingleLabelRuleEvaluationFactory(float64 l1RegularizationWeight,
                                                                                         float64 l2RegularizationWeight)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseSingleLabelRuleEvaluationFactory::create(
      const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        return std::make_unique<LabelWiseSingleLabelRuleEvaluation<DenseLabelWiseStatisticVector, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseSingleLabelRuleEvaluationFactory::create(
      const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWiseSingleLabelRuleEvaluation<DenseLabelWiseStatisticVector, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseLabelWiseStatisticVector>> LabelWiseSingleLabelRuleEvaluationFactory::create(
      const SparseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        return std::make_unique<
          LabelWiseSingleLabelRuleEvaluation<SparseLabelWiseStatisticVector, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseLabelWiseStatisticVector>> LabelWiseSingleLabelRuleEvaluationFactory::create(
      const SparseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWiseSingleLabelRuleEvaluation<SparseLabelWiseStatisticVector, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

}