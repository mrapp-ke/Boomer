#include "boosting/rule_evaluation/rule_evaluation_example_wise_complete_binned.hpp"

#include "rule_evaluation_example_wise_binned_common.hpp"

namespace boosting {

    ExampleWiseCompleteBinnedRuleEvaluationFactory::ExampleWiseCompleteBinnedRuleEvaluationFactory(
      float64 l1RegularizationWeight, float64 l2RegularizationWeight,
      std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr, const Blas& blas, const Lapack& lapack)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
          labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)), blas_(blas), lapack_(lapack) {}

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>>
      ExampleWiseCompleteBinnedRuleEvaluationFactory::create(const DenseExampleWiseStatisticVector& statisticVector,
                                                             const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseExampleWiseCompleteBinnedRuleEvaluation<CompleteIndexVector>>(
          indexVector, maxBins, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr), blas_,
          lapack_);
    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>>
      ExampleWiseCompleteBinnedRuleEvaluationFactory::create(const DenseExampleWiseStatisticVector& statisticVector,
                                                             const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseExampleWiseCompleteBinnedRuleEvaluation<PartialIndexVector>>(
          indexVector, maxBins, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr), blas_,
          lapack_);
    }

}
