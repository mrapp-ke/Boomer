/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "seco/rule_evaluation/rule_evaluation.hpp"
#include <memory>


namespace seco {

    /**
     * Defines an interface for all factories that allow to create instances of the type `IRuleEvaluation` that allow to
     * calculate the predictions of rules, as well as corresponding quality scores, based on label-wise confusion
     * matrices.
     */
    class ILabelWiseRuleEvaluationFactory {

        public:

            virtual ~ILabelWiseRuleEvaluationFactory() { };

            /**
             * Creates and returns a new object of type `IRuleEvaluation` that allows to calculate the predictions of
             * rules that predict for all available labels.
             *
             * @param indexVector   A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            virtual std::unique_ptr<IRuleEvaluation> create(const CompleteIndexVector& indexVector) const = 0;

            /**
             * Creates and returns a new object of type `IRuleEvaluation` that allows to calculate the predictions of
             * rules that predict for a subset of the available labels.
             *
             * @param indexVector   A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            virtual std::unique_ptr<IRuleEvaluation> create(const PartialIndexVector& indexVector) const = 0;

    };

}
