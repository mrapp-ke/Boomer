/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/rule_evaluation/rule_evaluation_example_wise.hpp"
#include "boosting/math/blas.hpp"
#include "boosting/math/lapack.hpp"


namespace boosting {

    /**
     * Allows to create instances of the class `RegularizedExampleWiseRuleEvaluation`.
     */
    class RegularizedExampleWiseRuleEvaluationFactory final : public IExampleWiseRuleEvaluationFactory {

        private:

            float64 l2RegularizationWeight_;

            std::shared_ptr<Blas> blasPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

        public:

            /**
             * @param l2RegularizationWeight The weight of the L2 regularization that is applied for calculating the
             *                               scores to be predicted by rules
             * @param blasPtr                A shared pointer to an object of type `Blas` that allows to execute
             *                               different BLAS routines
             * @param lapackPtr              A shared pointer to an object of type `Lapack` that allows to execute
             *                               different LAPACK routines
             */
            RegularizedExampleWiseRuleEvaluationFactory(float64 l2RegularizationWeight, std::shared_ptr<Blas> blasPtr,
                                                        std::shared_ptr<Lapack> lapackPtr);

            std::unique_ptr<IExampleWiseRuleEvaluation> create(const FullIndexVector& indexVector) const override;

            std::unique_ptr<IExampleWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const override;

    };

}
