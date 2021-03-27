/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/head_refinement/prediction_evaluated.hpp"
#include "common/statistics/statistics_subset.hpp"


/**
 * Defines an interface for all classes that allow to find the best head for a rule.
 */
class IHeadRefinement {

    public:

        virtual ~IHeadRefinement() { };

        /**
         * Finds the best head for a rule, given the predictions that are provided by a `IStatisticsSubset`.
         *
         * The given object of type `IStatisticsSubset` must have been prepared properly via calls to the function
         * `IStatisticsSubset#addToSubset`.
         *
         * @param bestHead          A pointer to an object of type `AbstractEvaluatedPrediction` that corresponds to the
         *                          best rule known so far (as found in the previous or current refinement iteration) or
         *                          a null pointer, if no such rule is available yet. The new head must be better than
         *                          this one, otherwise it is discarded
         * @param statisticsSubset  A reference to an object of type `IStatisticsSubset` to be used for calculating
         *                          predictions and corresponding quality scores
         * @param uncovered         False, if the rule for which the head should be found covers all statistics that
         *                          have been added to the `IStatisticsSubset` so far, True, if the rule covers all
         *                          statistics that have not been added yet
         * @param accumulated       False, if the rule covers all statistics that have been added since the
         *                          `IStatisticsSubset` has been reset for the last time, True, if the rule covers all
         *                          statistics that have been added so far
         * @return                  A pointer to an object of type `AbstractEvaluatedPrediction`, representing the head
         *                          that has been found or a null pointer if the head that has been found is not better
         *                          than `bestHead`
         */
        virtual const AbstractEvaluatedPrediction* findHead(const AbstractEvaluatedPrediction* bestHead,
                                                            IStatisticsSubset& statisticsSubset, bool uncovered,
                                                            bool accumulated) = 0;

        /**
         * Returns the best head that has been found by the function `findHead.
         *
         * @return An unique pointer to an object of type `AbstractEvaluatedPrediction`, representing the best head that
         *         has been found
         */
        virtual std::unique_ptr<AbstractEvaluatedPrediction> pollHead() = 0;

        /**
         * Calculates the optimal scores to be predicted by a rule, as well as the rule's overall quality score,
         * according to a `IStatisticsSubset`.
         *
         * The given object of type `IStatisticsSubset` must have been prepared properly via calls to the function
         * `IStatisticsSubset#addToSubset`.
         *
         * @param statisticsSubset  A reference to an object of type `IStatisticsSubset` to be used for calculating
         *                          predictions and corresponding quality scores
         * @param uncovered         False, if the rule for which the optimal scores should be calculated covers all
         *                          statistics that have been added to the `IStatisticsSubset` so far, True, if the rule
         *                          covers all statistics that have not been added yet
         * @param accumulated       False, if the rule covers all examples that have been added since the
         *                          `IStatisticsSubset` has been reset for the last time, True, if the rule covers all
         *                          examples that have been added so far
         * @return                  A reference to an object of type `IScoreVector` that stores the optimal scores to be
         *                          predicted by the rule, as well as its overall quality score
         */
        virtual const IScoreVector& calculatePrediction(IStatisticsSubset& statisticsSubset, bool uncovered,
                                                        bool accumulated) const = 0;

};
