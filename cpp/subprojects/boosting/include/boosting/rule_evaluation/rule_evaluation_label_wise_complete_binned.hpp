/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/rule_evaluation/rule_evaluation_label_wise.hpp"
#include "boosting/binning/label_binning.hpp"


namespace boosting {

    /**
     * Allows to create instances of the class `LabelWiseCompleteBinnedRuleEvaluationFactory`.
     */
    class LabelWiseCompleteBinnedRuleEvaluationFactory final : public ILabelWiseRuleEvaluationFactory {

        private:

            float64 l2RegularizationWeight_;

            std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr_;

        public:

            /**
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param labelBinningFactoryPtr    An unique pointer to an object of type `ILabelBinningFactory` that
             *                                  allows to create the implementation to be used to assign labels to bins
             */
            LabelWiseCompleteBinnedRuleEvaluationFactory(float64 l2RegularizationWeight,
                                                         std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr);

            std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> create(
                const DenseLabelWiseStatisticVector& statisticVector,
                const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> create(
                const DenseLabelWiseStatisticVector& statisticVector,
                const PartialIndexVector& indexVector) const override;

    };

}
