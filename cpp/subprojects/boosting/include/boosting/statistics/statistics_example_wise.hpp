/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics.hpp"
#include "boosting/rule_evaluation/rule_evaluation_example_wise.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that store gradients and Hessians that have been calculated according to a
     * differentiable loss-function that is applied example-wise.
     */
    class IExampleWiseStatistics : virtual public IStatistics {

        public:

            virtual ~IExampleWiseStatistics() { };

            /**
             * Sets the factory that allows to create instances of the class that is used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
             *
             * @param ruleEvaluationFactoryPtr A shared pointer to an object of type `IExampleWiseRuleFactoryEvaluation`
             *                                 to be set
             */
            virtual void setRuleEvaluationFactory(
                std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr) = 0;

    };

    /**
     * Defines an interface for all classes that allow to create new instances of the class `IExampleWiseStatistics`.
     */
    class IExampleWiseStatisticsFactory {

        public:

            virtual ~IExampleWiseStatisticsFactory() { };

            /**
             * Creates a new instance of the type `IExampleWiseStatistics`.
             *
             * @return An unique pointer to an object of type `IExampleWiseStatistics` that has been created
             */
            virtual std::unique_ptr<IExampleWiseStatistics> create() const = 0;

    };

}
