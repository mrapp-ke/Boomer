#include "boosting/rule_evaluation/rule_evaluation_example_wise_complete.hpp"

#include "rule_evaluation_example_wise_complete_common.hpp"

namespace boosting {

    ExampleWiseCompleteRuleEvaluationFactory::ExampleWiseCompleteRuleEvaluationFactory(float64 l1RegularizationWeight,
                                                                                       float64 l2RegularizationWeight,
                                                                                       const Blas& blas,
                                                                                       const Lapack& lapack)
        : l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight), blas_(blas),
          lapack_(lapack) {}

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseCompleteRuleEvaluationFactory::create(
      const DenseExampleWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        return std::make_unique<DenseExampleWiseCompleteRuleEvaluation<CompleteIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blas_, lapack_);
    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseCompleteRuleEvaluationFactory::create(
      const DenseExampleWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseExampleWiseCompleteRuleEvaluation<PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blas_, lapack_);
    }

}
