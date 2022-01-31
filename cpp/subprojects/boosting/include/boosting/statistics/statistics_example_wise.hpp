/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/statistics/statistics_label_wise.hpp"
#include "boosting/rule_evaluation/rule_evaluation_example_wise.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that store gradients and Hessians that have been calculated according to a
     * differentiable loss-function that is applied example-wise.
     *
     * @tparam ExampleWiseRuleEvaluationFactory The type of the classes that may be used for calculating the
     *                                          example-wise predictions, as well as corresponding quality scores, of
     *                                          rules
     * @tparam LabelWiseRuleEvaluationFactory   The type of the classes that may be used for calculating the label-wise
     *                                          predictions, as well as corresponding quality scores, of rules
     */
    template<typename ExampleWiseRuleEvaluationFactory, typename LabelWiseRuleEvaluationFactory>
    class IExampleWiseStatistics : virtual public IStatistics {

        public:

            virtual ~IExampleWiseStatistics() override { };

            /**
             * Sets the factory that allows to create instances of the class that is used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
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
             *                              the predictions, as well as corresponding quality scores of rules
             * @param numThreads            The number of threads that should be used to convert the statistics for
             *                              individual examples in parallel
             * @return                      An unique pointer to an object of type `ILabelWiseStatistics` that has been
             *                              created
             */
            virtual std::unique_ptr<ILabelWiseStatistics<LabelWiseRuleEvaluationFactory>> toLabelWiseStatistics(
                const LabelWiseRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads) = 0;

    };

}
