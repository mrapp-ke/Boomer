/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/binning/label_binning.hpp"
#include "boosting/math/blas.hpp"
#include "boosting/math/lapack.hpp"
#include "boosting/rule_evaluation/rule_evaluation_example_wise.hpp"

namespace boosting {

    /**
     * Allows to create instances of the class `IExampleWiseRuleEvaluationFactory` that allow to calculate the
     * predictions of partial rules, which predict for a predefined number of labels, using gradient-based label
     * binning.
     */
    class ExampleWiseFixedPartialBinnedRuleEvaluationFactory final : public IExampleWiseRuleEvaluationFactory {
        private:

            const float32 labelRatio_;

            const uint32 minLabels_;

            const uint32 maxLabels_;

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

            const std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr_;

            const Blas& blas_;

            const Lapack& lapack_;

        public:

            /**
             * @param labelRatio                A percentage that specifies for how many labels the rule heads should
             *                                  predict, e.g., if 100 labels are available, a percentage of 0.5 means
             *                                  that the rule heads predict for a subset of `ceil(0.5 * 100) = 50`
             *                                  labels. Must be in (0, 1)
             * @param minLabels                 The minimum number of labels for which the rule heads should predict.
             *                                  Must be at least 2
             * @param maxLabels                 The maximum number of labels for which the rule heads should predict.
             *                                  Must be at least `minLabels` or 0, if the maximum number of labels
             *                                  should not be restricted
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param labelBinningFactoryPtr    An unique pointer to an object of type `ILabelBinningFactory` that
             *                                  allows to create the implementation to be used to assign labels to bins
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    An reference to an object of type `Lapack` that allows to execute BLAS
             *                                  routines
             */
            ExampleWiseFixedPartialBinnedRuleEvaluationFactory(
              float32 labelRatio, uint32 minLabels, uint32 maxLabels, float64 l1RegularizationWeight,
              float64 l2RegularizationWeight, std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr,
              const Blas& blas, const Lapack& lapack);

            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> create(
              const DenseExampleWiseStatisticVector& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> create(
              const DenseExampleWiseStatisticVector& statisticVector,
              const PartialIndexVector& indexVector) const override;
    };

}
