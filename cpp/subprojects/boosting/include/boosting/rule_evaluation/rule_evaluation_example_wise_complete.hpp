/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/math/blas.hpp"
#include "boosting/math/lapack.hpp"
#include "boosting/rule_evaluation/rule_evaluation_example_wise.hpp"

namespace boosting {

    /**
     * Allows to create instances of the class `IExampleWiseRuleEvaluationFactory` that allow to calculate the
     * predictions of complete rules, which predict for all available labels.
     */
    class ExampleWiseCompleteRuleEvaluationFactory final : public IExampleWiseRuleEvaluationFactory {
        private:

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

            const Blas& blas_;

            const Lapack& lapack_;

        public:

            /**
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    An reference to an object of type `Lapack` that allows to execute BLAS
             *                                  routines
             */
            ExampleWiseCompleteRuleEvaluationFactory(float64 l1RegularizationWeight, float64 l2RegularizationWeight,
                                                     const Blas& blas, const Lapack& lapack);

            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> create(
              const DenseExampleWiseStatisticVector& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> create(
              const DenseExampleWiseStatisticVector& statisticVector,
              const PartialIndexVector& indexVector) const override;
    };

}
