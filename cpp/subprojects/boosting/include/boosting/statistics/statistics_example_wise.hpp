/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/rule_evaluation/rule_evaluation_example_wise.hpp"
#include "boosting/statistics/statistics_label_wise.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that store gradients and Hessians that have been calculated according to a
     * differentiable loss-function that is applied example-wise.
     *
     * @tparam ExampleWiseRuleEvaluationFactory The type of the classes that may be used for calculating the
     *                                          example-wise predictions of rules, as well as their overall quality
     * @tparam LabelWiseRuleEvaluationFactory   The type of the classes that may be used for calculating the label-wise
     *                                          predictions of rules, as well as their overall quality
     */
    template<typename ExampleWiseRuleEvaluationFactory, typename LabelWiseRuleEvaluationFactory>
    class IExampleWiseStatistics : virtual public IBoostingStatistics {
        public:

            virtual ~IExampleWiseStatistics() override {};

            /**
             * Sets the factory that allows to create instances of the class that is used for calculating the
             * predictions of rules, as well as their overall quality.
             *
             * @param ruleEvaluationFactory A reference to an object of template type `ExampleWiseRuleEvaluationFactory`
             *                              to be set
             */
            virtual void setRuleEvaluationFactory(const ExampleWiseRuleEvaluationFactory& ruleEvaluationFactory) = 0;

            /**
             * Creates and returns an instance of type `ILabelWiseStatistics` from the gradients and Hessians that are
             * stored by this object.
             *
             * @param ruleEvaluationFactory A reference to an object of template type `LabelWiseRuleEvaluationFactory`
             *                              that allows to create instances of the class that is used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param numThreads            The number of threads that should be used to convert the statistics for
             *                              individual examples in parallel
             * @return                      An unique pointer to an object of type `ILabelWiseStatistics` that has been
             *                              created
             */
            virtual std::unique_ptr<ILabelWiseStatistics<LabelWiseRuleEvaluationFactory>> toLabelWiseStatistics(
              const LabelWiseRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads) = 0;
    };

}
