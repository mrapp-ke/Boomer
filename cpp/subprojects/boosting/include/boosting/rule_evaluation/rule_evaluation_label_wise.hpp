/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "boosting/data/statistic_vector_dense_label_wise.hpp"
#include "boosting/rule_evaluation/rule_evaluation.hpp"
#include <memory>


namespace boosting {

    /**
     * Defines an interface for all factories that allow to create instances of the type `IRuleEvaluation` that allow to
     * calculate the predictions of rules, based on the gradients and Hessians that have been calculated according to a
     * loss function that is applied label-wise.
     */
    class ILabelWiseRuleEvaluationFactory {

        public:

            virtual ~ILabelWiseRuleEvaluationFactory() { };

            /**
             * Creates a new instance of the class `IRuleEvaluation` that allows to calculate the predictions of rules
             * that predict for all available labels, based on the gradients and Hessians that are stored by a
             * `DenseLabelWiseStatisticVector`.
             *
             * @param indexVector   A reference to an object of the type `CompleteIndexVector` that provides access to
             *                      the indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            virtual std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> createDense(
                const CompleteIndexVector& indexVector) const = 0;

            /**
             * Creates a new instance of the class `IRuleEvaluation` that allows to calculate the predictions of rules
             * that predict for a subset of the available labels, based on the gradients and Hessians that are stored by
             * a `DenseLabelWiseStatisticVector`.
             *
             * @param indexVector   A reference to an object of the type `PartialIndexVector` that provides access to
             *                      the indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            virtual std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> createDense(
                const PartialIndexVector& indexVector) const = 0;

    };

}
