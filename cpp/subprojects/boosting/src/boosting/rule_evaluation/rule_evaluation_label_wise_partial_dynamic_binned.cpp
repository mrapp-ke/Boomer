#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_dynamic_binned.hpp"

#include "rule_evaluation_label_wise_binned_common.hpp"
#include "rule_evaluation_label_wise_partial_dynamic_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a subset of the available labels that is
     * determined dynamically, as well as their overall quality, based on the gradients and Hessians that are stored by
     * a vector using L1 and L2 regularization. The labels are assigned to bins based on the gradients and Hessians.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the labels for which predictions should
     *                          be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class LabelWiseDynamicPartialBinnedRuleEvaluation final
        : public AbstractLabelWiseBinnedRuleEvaluation<StatisticVector, PartialIndexVector> {
        private:

            const IndexVector& labelIndices_;

            const std::unique_ptr<PartialIndexVector> indexVectorPtr_;

            const float64 threshold_;

            const float64 exponent_;

        protected:

            uint32 calculateLabelWiseCriteria(const StatisticVector& statisticVector, float64* criteria,
                                              uint32 numCriteria, float64 l1RegularizationWeight,
                                              float64 l2RegularizationWeight) override {
                uint32 numElements = statisticVector.getNumElements();
                typename StatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                const std::pair<float64, float64> pair =
                  getMinAndMaxScore(statisticIterator, numElements, l1RegularizationWeight, l2RegularizationWeight);
                float64 minAbsScore = pair.first;
                float64 threshold = calculateThreshold(minAbsScore, pair.second, threshold_, exponent_);
                PartialIndexVector::iterator indexIterator = indexVectorPtr_->begin();
                typename IndexVector::const_iterator labelIndexIterator = labelIndices_.cbegin();
                uint32 n = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    float64 score = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight,
                                                            l2RegularizationWeight);

                    if (calculateWeightedScore(score, minAbsScore, exponent_) > threshold) {
                        indexIterator[n] = labelIndexIterator[i];
                        criteria[n] = score;
                        n++;
                    }
                }

                indexVectorPtr_->setNumElements(n, false);
                return n;
            }

        public:

            /**
             * @param labelIndices              A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the labels for which the rules may predict
             * @param indexVectorPtr            An unique pointer to an object of type `PartialIndexVector` that stores
             *                                  the indices of the labels for which a rule predicts
             * @param threshold                 A threshold that affects for how many labels the rule heads should
             *                                  predict
             * @param exponent                  An exponent that is used to weigh the estimated predictive quality for
             *                                  individual labels
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             */
            LabelWiseDynamicPartialBinnedRuleEvaluation(const IndexVector& labelIndices,
                                                        std::unique_ptr<PartialIndexVector> indexVectorPtr,
                                                        float32 threshold, float32 exponent,
                                                        float64 l1RegularizationWeight, float64 l2RegularizationWeight,
                                                        std::unique_ptr<ILabelBinning> binningPtr)
                : AbstractLabelWiseBinnedRuleEvaluation<StatisticVector, PartialIndexVector>(
                  *indexVectorPtr, true, l1RegularizationWeight, l2RegularizationWeight, std::move(binningPtr)),
                  labelIndices_(labelIndices), indexVectorPtr_(std::move(indexVectorPtr)), threshold_(1.0 - threshold),
                  exponent_(exponent) {}
    };

    LabelWiseDynamicPartialBinnedRuleEvaluationFactory::LabelWiseDynamicPartialBinnedRuleEvaluationFactory(
      float32 threshold, float32 exponent, float64 l1RegularizationWeight, float64 l2RegularizationWeight,
      std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr)
        : threshold_(threshold), exponent_(exponent), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight), labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)) {}

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>>
      LabelWiseDynamicPartialBinnedRuleEvaluationFactory::create(const DenseLabelWiseStatisticVector& statisticVector,
                                                                 const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
          std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<
          LabelWiseDynamicPartialBinnedRuleEvaluation<DenseLabelWiseStatisticVector, CompleteIndexVector>>(
          indexVector, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>>
      LabelWiseDynamicPartialBinnedRuleEvaluationFactory::create(const DenseLabelWiseStatisticVector& statisticVector,
                                                                 const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<
          LabelWiseCompleteBinnedRuleEvaluation<DenseLabelWiseStatisticVector, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<SparseLabelWiseStatisticVector>>
      LabelWiseDynamicPartialBinnedRuleEvaluationFactory::create(const SparseLabelWiseStatisticVector& statisticVector,
                                                                 const CompleteIndexVector& indexVector) const {
        std::unique_ptr<PartialIndexVector> indexVectorPtr =
          std::make_unique<PartialIndexVector>(indexVector.getNumElements());
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<
          LabelWiseDynamicPartialBinnedRuleEvaluation<SparseLabelWiseStatisticVector, CompleteIndexVector>>(
          indexVector, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<SparseLabelWiseStatisticVector>>
      LabelWiseDynamicPartialBinnedRuleEvaluationFactory::create(const SparseLabelWiseStatisticVector& statisticVector,
                                                                 const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<
          LabelWiseCompleteBinnedRuleEvaluation<SparseLabelWiseStatisticVector, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

}
