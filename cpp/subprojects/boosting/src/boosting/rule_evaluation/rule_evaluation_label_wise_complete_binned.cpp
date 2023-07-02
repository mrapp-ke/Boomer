#include "boosting/rule_evaluation/rule_evaluation_label_wise_complete_binned.hpp"

#include "rule_evaluation_label_wise_binned_common.hpp"

namespace boosting {

    LabelWiseCompleteBinnedRuleEvaluationFactory::LabelWiseCompleteBinnedRuleEvaluationFactory(
      float64 l1RegularizationWeight, float64 l2RegularizationWeight,
      std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)) {}

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>>
      LabelWiseCompleteBinnedRuleEvaluationFactory::create(const DenseLabelWiseStatisticVector& statisticVector,
                                                           const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<
          LabelWiseCompleteBinnedRuleEvaluation<DenseLabelWiseStatisticVector, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>>
      LabelWiseCompleteBinnedRuleEvaluationFactory::create(const DenseLabelWiseStatisticVector& statisticVector,
                                                           const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<
          LabelWiseCompleteBinnedRuleEvaluation<DenseLabelWiseStatisticVector, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

}
