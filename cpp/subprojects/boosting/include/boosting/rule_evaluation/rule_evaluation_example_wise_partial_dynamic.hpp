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
     * predictions of partial rules, which predict for a subset of the available that is determined dynamically.
     */
    class ExampleWiseDynamicPartialRuleEvaluationFactory final : public IExampleWiseRuleEvaluationFactory {
        private:

            const float32 threshold_;

            const float32 exponent_;

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

            const Blas& blas_;

            const Lapack& lapack_;

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
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    An reference to an object of type `Lapack` that allows to execute BLAS
             *                                  routines
             */
            ExampleWiseDynamicPartialRuleEvaluationFactory(float32 threshold, float32 exponent,
                                                           float64 l1RegularizationWeight,
                                                           float64 l2RegularizationWeight, const Blas& blas,
                                                           const Lapack& lapack);

            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> create(
              const DenseExampleWiseStatisticVector& statisticVector,
              const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> create(
              const DenseExampleWiseStatisticVector& statisticVector,
              const PartialIndexVector& indexVector) const override;
    };

}
