/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/statistics/statistics_label_wise.hpp"
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
             * @param ruleEvaluationFactory A reference to an object of type `IExampleWiseRuleFactoryEvaluation` to be
             *                              set
             */
            virtual void setRuleEvaluationFactory(const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory) = 0;

            /**
             * Creates and returns an instance of type `ILabelWiseStatistics` from the gradients and Hessians that are
             * stored by this object.
             *
             * @param ruleEvaluationFactory A reference to an object of type `ILabelWiseRuleEvaluationFactory` that
             *                              allows to create instances of the class that is used for calculating the
             *                              predictions, as well as corresponding quality scores of rules
             * @param numThreads            The number of threads that should be used to convert the statistics for
             *                              individual examples in parallel
             * @return                      An unique pointer to an object of type `ILabelWiseStatistics` that has been
             *                              created
             */
            virtual std::unique_ptr<ILabelWiseStatistics> toLabelWiseStatistics(
                const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads) = 0;

    };

    /**
     * Defines an interface for all classes that allow to create new instances of the class `IExampleWiseStatistics`.
     */
    class IExampleWiseStatisticsFactory {

        public:

            virtual ~IExampleWiseStatisticsFactory() { };

            /**
             * Creates a new instance of the type `IExampleWiseStatistics`, based on a matrix that provides random
             * access to the labels of the training examples.
             *
             * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` that provides random
             *                      access to the labels of the training examples
             * @return              An unique pointer to an object of type `IExampleWiseStatistics` that has been
             *                      created
             */
            virtual std::unique_ptr<IExampleWiseStatistics> create(const CContiguousLabelMatrix& labelMatrix) const = 0;

            /**
             * Creates a new instance of the type `IExampleWiseStatistics`, based on a matrix that provides row-wise
             * access to the labels of the training examples.
             *
             * @param labelMatrix   A reference to an object of type `CsrLabelMatrix` that provides row-wise access to
             *                      the labels of the training examples
             * @return              An unique pointer to an object of type `IExampleWiseStatistics` that has been
             *                      created
             */
            virtual std::unique_ptr<IExampleWiseStatistics> create(const CsrLabelMatrix& labelMatrix) const = 0;

    };

}
