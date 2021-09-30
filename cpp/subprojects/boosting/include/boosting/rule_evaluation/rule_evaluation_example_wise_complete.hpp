/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/rule_evaluation/rule_evaluation_example_wise.hpp"
#include "boosting/math/blas.hpp"
#include "boosting/math/lapack.hpp"


namespace boosting {

    /**
     * Allows to create instances of the class `ExampleWiseCompleteRuleEvaluationFactory`.
     */
    class ExampleWiseCompleteRuleEvaluationFactory final : public IExampleWiseRuleEvaluationFactory {

        private:

            float64 l2RegularizationWeight_;

            std::unique_ptr<Blas> blasPtr_;

            std::unique_ptr<Lapack> lapackPtr_;

        public:

            /**
             * @param l2RegularizationWeight The weight of the L2 regularization that is applied for calculating the
             *                               scores to be predicted by rules
             * @param blasPtr                An unique pointer to an object of type `Blas` that allows to execute
             *                               different BLAS routines
             * @param lapackPtr              An unique pointer to an object of type `Lapack` that allows to execute
             *                               different LAPACK routines
             */
            ExampleWiseCompleteRuleEvaluationFactory(float64 l2RegularizationWeight, std::unique_ptr<Blas> blasPtr,
                                                     std::unique_ptr<Lapack> lapackPtr);

            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> create(
                const DenseExampleWiseStatisticVector& statisticVector,
                const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> create(
                const DenseExampleWiseStatisticVector& statisticVector,
                const PartialIndexVector& indexVector) const override;

    };

}
