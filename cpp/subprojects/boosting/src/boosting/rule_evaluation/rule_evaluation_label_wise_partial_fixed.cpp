#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_fixed.hpp"

#include "rule_evaluation_label_wise_complete_common.hpp"
#include "rule_evaluation_label_wise_partial_fixed_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a predefined number of labels, as well as
     * their overall quality, based on the gradients and Hessians that are stored by a vector using L1 and L2
     * regularization.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the labels for which predictions should
     *                          be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class LabelWiseFixedPartialRuleEvaluation final : public IRuleEvaluation<StatisticVector> {
        private:

            const IndexVector& labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

            SparseArrayVector<float64> tmpVector_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the labels for which the rules may predict
             * @param numPredictions            The number of labels for which the rules should predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            LabelWiseFixedPartialRuleEvaluation(const IndexVector& labelIndices, uint32 numPredictions,
                                                float64 l1RegularizationWeight, float64 l2RegularizationWeight)
                : labelIndices_(labelIndices), indexVector_(PartialIndexVector(numPredictions)),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_, false)),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
                  tmpVector_(SparseArrayVector<float64>(labelIndices.getNumElements())) {}

            const IScoreVector& calculateScores(StatisticVector& statisticVector) override {
                uint32 numElements = statisticVector.getNumElements();
                uint32 numPredictions = indexVector_.getNumElements();
                typename StatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                SparseArrayVector<float64>::iterator tmpIterator = tmpVector_.begin();
                sortLabelWiseScores(tmpIterator, statisticIterator, numElements, numPredictions,
                                    l1RegularizationWeight_, l2RegularizationWeight_);
                PartialIndexVector::iterator indexIterator = indexVector_.begin();
                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                typename IndexVector::const_iterator labelIndexIterator = labelIndices_.cbegin();
                float64 quality = 0;

                for (uint32 i = 0; i < numPredictions; i++) {
                    const IndexedValue<float64>& entry = tmpIterator[i];
                    uint32 index = entry.index;
                    float64 predictedScore = entry.value;
                    indexIterator[i] = labelIndexIterator[index];
                    scoreIterator[i] = predictedScore;
                    const Tuple<float64>& tuple = statisticIterator[index];
                    quality += calculateLabelWiseQuality(predictedScore, tuple.first, tuple.second,
                                                         l1RegularizationWeight_, l2RegularizationWeight_);
                }

                scoreVector_.quality = quality;
                return scoreVector_;
            }
    };

    LabelWiseFixedPartialRuleEvaluationFactory::LabelWiseFixedPartialRuleEvaluationFactory(
      float32 labelRatio, uint32 minLabels, uint32 maxLabels, float64 l1RegularizationWeight,
      float64 l2RegularizationWeight)
        : labelRatio_(labelRatio), minLabels_(minLabels), maxLabels_(maxLabels),
          l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseFixedPartialRuleEvaluationFactory::create(
      const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        uint32 numPredictions =
          calculateBoundedFraction(indexVector.getNumElements(), labelRatio_, minLabels_, maxLabels_);
        return std::make_unique<
          LabelWiseFixedPartialRuleEvaluation<DenseLabelWiseStatisticVector, CompleteIndexVector>>(
          indexVector, numPredictions, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseFixedPartialRuleEvaluationFactory::create(
      const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWiseCompleteRuleEvaluation<DenseLabelWiseStatisticVector, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseLabelWiseStatisticVector>> LabelWiseFixedPartialRuleEvaluationFactory::create(
      const SparseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        uint32 numPredictions =
          calculateBoundedFraction(indexVector.getNumElements(), labelRatio_, minLabels_, maxLabels_);
        return std::make_unique<
          LabelWiseFixedPartialRuleEvaluation<SparseLabelWiseStatisticVector, CompleteIndexVector>>(
          indexVector, numPredictions, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<SparseLabelWiseStatisticVector>> LabelWiseFixedPartialRuleEvaluationFactory::create(
      const SparseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWiseCompleteRuleEvaluation<SparseLabelWiseStatisticVector, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

}
