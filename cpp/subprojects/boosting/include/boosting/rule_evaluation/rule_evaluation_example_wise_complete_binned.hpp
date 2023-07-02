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
     * predictions of complete rules, which predict for all available labels, using gradient-based label binning.
     */
    class ExampleWiseCompleteBinnedRuleEvaluationFactory final : public IExampleWiseRuleEvaluationFactory {
        private:

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

            const std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr_;

            const Blas& blas_;

            const Lapack& lapack_;

        public:

            /**
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param labelBinningFactoryPtr    An unique pointer to an object of type `ILabelBinningFactory` that
             *                                  allows to create the implementation to be used to assign labels to bins
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute LAPACK
             *                                  routines
             */
            ExampleWiseCompleteBinnedRuleEvaluationFactory(float64 l1RegularizationWeight,
                                                           float64 l2RegularizationWeight,
                                                           std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr,
                                                           const Blas& blas, const Lapack& lapack);

            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> create(
              const DenseExampleWiseStatisticVector& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> create(
              const DenseExampleWiseStatisticVector& statisticVector,
              const PartialIndexVector& indexVector) const override;
    };

}
