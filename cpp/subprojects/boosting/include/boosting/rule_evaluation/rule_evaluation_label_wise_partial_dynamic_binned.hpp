/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/binning/label_binning.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_sparse.hpp"

namespace boosting {

    /**
     * Allows to create instances of the class `ISparseLabelWiseRuleEvaluationFactory` that allow to calculate the
     * predictions of partial rules, which predict for a subset of the available labels that is determined dynamically,
     * using gradient-based label binning.
     */
    class LabelWiseDynamicPartialBinnedRuleEvaluationFactory final : public ISparseLabelWiseRuleEvaluationFactory {
        private:

            const float32 threshold_;

            const float32 exponent_;

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

            const std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr_;

        public:

            /**
             * @param threshold                 A threshold that affects for how many labels the rule heads should
             *                                  predict. A smaller threshold results in less labels being selected. A
             *                                  greater threshold results in more labels being selected. E.g., a
             *                                  threshold of 0.2 means that a rule will only predict for a label if the
             *                                  estimated predictive quality `q` for this particular label satisfies the
             *                                  inequality `q^exponent > q_best^exponent * (1 - 0.2)`, where `q_best` is
             *                                  the best quality among all labels. Must be in (0, 1)
             * @param exponent                  An exponent that should be used to weigh the estimated predictive
             *                                  quality for individual labels. E.g., an exponent of 2 means that the
             *                                  estimated predictive quality `q` for a particular label is weighed as
             *                                  `q^2`. Must be at least 1
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param labelBinningFactoryPtr    An unique pointer to an object of type `ILabelBinningFactory` that
             *                                  allows to create the implementation to be used to assign labels to bins
             */
            LabelWiseDynamicPartialBinnedRuleEvaluationFactory(
              float32 threshold, float32 exponent, float64 l1RegularizationWeight, float64 l2RegularizationWeight,
              std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr);

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
