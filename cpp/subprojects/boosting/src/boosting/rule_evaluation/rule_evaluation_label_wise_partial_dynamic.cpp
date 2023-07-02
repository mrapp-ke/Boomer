#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_dynamic.hpp"

#include "rule_evaluation_label_wise_complete_common.hpp"
#include "rule_evaluation_label_wise_partial_dynamic_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules, which predict for a subset of the available labels that is
     * determined dynamically, as well as their overall quality, based on the gradients and Hessians that are stored by
     * a vector using L1 and L2 regularization.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the labels for which predictions should
     *                          be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class LabelWiseDynamicPartialRuleEvaluation final : public IRuleEvaluation<StatisticVector> {
        private:

            const IndexVector& labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            const float64 threshold_;

            const float64 exponent_;

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the labels for which the rules may predict
             * @param threshold                 A threshold that affects for how many labels the rule heads should
             *                                  predict
             * @param exponent                  An exponent that is used to weigh that estimated predictive quality for
             *                                  individual labels
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            LabelWiseDynamicPartialRuleEvaluation(const IndexVector& labelIndices, float32 threshold, float32 exponent,
                                                  float64 l1RegularizationWeight, float64 l2RegularizationWeight)
                : labelIndices_(labelIndices), indexVector_(PartialIndexVector(labelIndices.getNumElements())),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_, true)), threshold_(1.0 - threshold),
                  exponent_(exponent), l1RegularizationWeight_(l1RegularizationWeight),
                  l2RegularizationWeight_(l2RegularizationWeight) {}

            const IScoreVector& calculateScores(StatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                typename StatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                const std::pair<float64, float64> pair =
                  getMinAndMaxScore(statisticIterator, numElements, l1RegularizationWeight_, l2RegularizationWeight_);
                float64 minAbsScore = pair.first;
                float64 threshold = calculateThreshold(minAbsScore, pair.second, threshold_, exponent_);
                PartialIndexVector::iterator indexIterator = indexVector_.begin();
                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                typename IndexVector::const_iterator labelIndexIterator = labelIndices_.cbegin();
                float64 quality = 0;
                uint32 n = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    float64 score = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight_,
                                                            l2RegularizationWeight_);

                    if (calculateWeightedScore(score, minAbsScore, exponent_) > threshold) {
                        indexIterator[n] = labelIndexIterator[i];
                        scoreIterator[n] = score;
                        quality += calculateLabelWiseQuality(score, tuple.first, tuple.second, l1RegularizationWeight_,
                                                             l2RegularizationWeight_);
                        n++;
                    }
                }

                indexVector_.setNumElements(n, false);
                scoreVector_.quality = quality;
                return scoreVector_;
            }
    };

    LabelWiseDynamicPartialRuleEvaluationFactory::LabelWiseDynamicPartialRuleEvaluationFactory(
      float32 threshold, float32 exponent, float64 l1RegularizationWeight, float64 l2RegularizationWeight)
        : threshold_(threshold), exponent_(exponent), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight) {}

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>>
      LabelWiseDynamicPartialRuleEvaluationFactory::create(const DenseLabelWiseStatisticVector& statisticVector,
                                                           const CompleteIndexVector& indexVector) const {
        return std::make_unique<
          LabelWiseDynamicPartialRuleEvaluation<DenseLabelWiseStatisticVector, CompleteIndexVector>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>>
      LabelWiseDynamicPartialRuleEvaluationFactory::create(const DenseLabelWiseStatisticVector& statisticVector,
                                                           const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWiseCompleteRuleEvaluation<DenseLabelWiseStatisticVector, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseLabelWiseStatisticVector>>
      LabelWiseDynamicPartialRuleEvaluationFactory::create(const SparseLabelWiseStatisticVector& statisticVector,
                                                           const CompleteIndexVector& indexVector) const {
        return std::make_unique<
          LabelWiseDynamicPartialRuleEvaluation<SparseLabelWiseStatisticVector, CompleteIndexVector>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseLabelWiseStatisticVector>>
      LabelWiseDynamicPartialRuleEvaluationFactory::create(const SparseLabelWiseStatisticVector& statisticVector,
                                                           const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWiseCompleteRuleEvaluation<SparseLabelWiseStatisticVector, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

}
