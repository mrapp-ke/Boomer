/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/binning/label_binning.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise.hpp"

namespace boosting {

    /**
     * Allows to create instances of the class `ILabelWiseRuleEvaluationFactory` that allow to calculate the predictions
     * of complete rules, which predict for all available labels, using gradient-based label binning.
     */
    class LabelWiseCompleteBinnedRuleEvaluationFactory final : public ILabelWiseRuleEvaluationFactory {
        private:

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

            const std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr_;

        public:

            /**
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param labelBinningFactoryPtr    An unique pointer to an object of type `ILabelBinningFactory` that
             *                                  allows to create the implementation to be used to assign labels to bins
             */
            LabelWiseCompleteBinnedRuleEvaluationFactory(float64 l1RegularizationWeight, float64 l2RegularizationWeight,
                                                         std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr);

            std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> create(
              const DenseLabelWiseStatisticVector& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> create(
              const DenseLabelWiseStatisticVector& statisticVector,
              const PartialIndexVector& indexVector) const override;
    };

}
