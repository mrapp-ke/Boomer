/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/rule_evaluation/rule_evaluation_label_wise.hpp"


namespace boosting {

    /**
     * Allows to create instances of the class `RegularizedLabelWiseRuleEvaluation`.
     */
    class RegularizedLabelWiseRuleEvaluationFactory final : public ILabelWiseRuleEvaluationFactory {

        private:

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param l2RegularizationWeight The weight of the L2 regularization that is applied for calculating the
             *                               scores to be predicted by rules
             */
            RegularizedLabelWiseRuleEvaluationFactory(float64 l2RegularizationWeight);

            std::unique_ptr<ILabelWiseRuleEvaluation> create(const FullIndexVector& indexVector) const override;

            std::unique_ptr<ILabelWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const override;

    };

}
