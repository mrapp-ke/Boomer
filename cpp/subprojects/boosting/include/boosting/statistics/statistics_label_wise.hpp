/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/statistics/statistics.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that store gradients and Hessians that have been calculated according to a
     * differentiable loss function that is applied label-wise.
     *
     * @tparam RuleEvaluationFactory The type of the classes that may be used for calculating the predictions, as well
     *                               as corresponding quality scores, of rules
     */
    template<typename RuleEvaluationFactory>
    class ILabelWiseStatistics : virtual public IStatistics {

        public:

            virtual ~ILabelWiseStatistics() override { };

            /**
             * Sets the factory that allows to create instances of the class that is used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
             *
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` to be set
             */
            virtual void setRuleEvaluationFactory(const RuleEvaluationFactory& ruleEvaluationFactory) = 0;

    };

}
