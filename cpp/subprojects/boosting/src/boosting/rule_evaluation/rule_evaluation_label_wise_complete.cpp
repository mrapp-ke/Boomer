#include "boosting/rule_evaluation/rule_evaluation_label_wise_complete.hpp"

#include "rule_evaluation_label_wise_complete_common.hpp"

namespace boosting {

    LabelWiseCompleteRuleEvaluationFactory::LabelWiseCompleteRuleEvaluationFactory(float64 l1RegularizationWeight,
                                                                                   float64 l2RegularizationWeight)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight) {}

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseCompleteRuleEvaluationFactory::create(
      const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        return std::make_unique<LabelWiseCompleteRuleEvaluation<DenseLabelWiseStatisticVector, CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseCompleteRuleEvaluationFactory::create(
      const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWiseCompleteRuleEvaluation<DenseLabelWiseStatisticVector, PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_);
    }

}