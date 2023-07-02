/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/rule_evaluation/rule_evaluation_label_wise_sparse.hpp"

namespace boosting {

    /**
     * Allows to create instances of the class `ISparseLabelWiseRuleEvaluationFactory` that allow to calculate the
     * predictions of single-label rules, which predict for a single label.
     */
    class LabelWiseSingleLabelRuleEvaluationFactory final : public ISparseLabelWiseRuleEvaluationFactory {
        private:

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

        public:

            /**
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            LabelWiseSingleLabelRuleEvaluationFactory(float64 l1RegularizationWeight, float64 l2RegularizationWeight);

            std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> create(
              const DenseLabelWiseStatisticVector& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> create(
              const DenseLabelWiseStatisticVector& statisticVector,
              const PartialIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<SparseLabelWiseStatisticVector>> create(
              const SparseLabelWiseStatisticVector& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<SparseLabelWiseStatisticVector>> create(
              const SparseLabelWiseStatisticVector& statisticVector,
              const PartialIndexVector& indexVector) const override;
    };

}
