/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that store gradients and Hessians that have been calculated according to a
     * differentiable loss function that is applied label-wise.
     */
    class ILabelWiseStatistics : virtual public IStatistics {

        public:

            virtual ~ILabelWiseStatistics() { };

            /**
             * Sets the factory that allows to create instances of the class that is used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
             *
             * @param ruleEvaluationFactoryPtr A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                 to be set
             */
            virtual void setRuleEvaluationFactory(
                std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr) = 0;

    };

    /**
     * Defines an interface for all classes that allow to create new instances of the type `ILabelWiseStatistics`.
     */
    class ILabelWiseStatisticsFactory {

        public:

            virtual ~ILabelWiseStatisticsFactory() { };

            /**
             * Creates a new instance of the type `ILabelWiseStatistics`.
             *
             * @return An unique pointer to an object of type `ILabelWiseStatistics` that has been created
             */
            virtual std::unique_ptr<ILabelWiseStatistics> create() const = 0;

    };

}
