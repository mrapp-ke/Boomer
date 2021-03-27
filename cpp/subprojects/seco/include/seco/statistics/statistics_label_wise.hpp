/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "seco/statistics/statistics_coverage.hpp"
#include "seco/rule_evaluation/rule_evaluation_label_wise.hpp"
#include <memory>


namespace seco {

    /**
     * Defines an interface for all classes that allow to store the elements of confusion matrices that are computed
     * independently for each label.
     */
    class ILabelWiseStatistics : public ICoverageStatistics {

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
     * Defines an interface for all classes that allow to create new instances of the class ILabelWiseStatistics`.
     */
    class ILabelWiseStatisticsFactory {

        public:

            virtual ~ILabelWiseStatisticsFactory() { };

            /**
             * Creates a new instance of the class `ILabelWiseStatistics`.
             *
             * @return An unique pointer to an object of type `ILabelWiseStatistics` that has been created
             */
            virtual std::unique_ptr<ILabelWiseStatistics> create() const = 0;

    };

}
