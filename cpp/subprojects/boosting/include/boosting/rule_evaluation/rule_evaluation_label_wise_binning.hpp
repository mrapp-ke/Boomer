/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/rule_evaluation/rule_evaluation_label_wise.hpp"


namespace boosting {

    /**
     * Allows to create instances of the class `BinningLabelWiseRuleEvaluation` that uses equal-width binning.
     */
    class EqualWidthBinningLabelWiseRuleEvaluationFactory final : public ILabelWiseRuleEvaluationFactory {

        private:

            float64 l2RegularizationWeight_;

            float32 binRatio_;

            uint32 minBins_;

            uint32 maxBins_;

        public:

            /**
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binRatio                  A percentage that specifies how many bins should be used to assign
             *                                  labels to
             * @param minBins                   The minimum number of bins to be used to assign labels to
             * @param maxBins                   The maximum number of bins to be used to assign labels to
             */
            EqualWidthBinningLabelWiseRuleEvaluationFactory(float64 l2RegularizationWeight, float32 binRatio,
                                                            uint32 minBins, uint32 maxBins);

            std::unique_ptr<ILabelWiseRuleEvaluation> create(const FullIndexVector& indexVector) const override;

            std::unique_ptr<ILabelWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const override;

    };

}
