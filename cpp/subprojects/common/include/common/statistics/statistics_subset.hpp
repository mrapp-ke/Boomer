/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_evaluation/score_vector.hpp"


/**
 * Defines an interface for all classes that provide access to a subset of the statistics that are stored by an instance
 * of the class `IHistogram` or `IStatistics` and allows to calculate the scores to be predicted by rules that cover
 * such a subset.
 */
class IStatisticsSubset {

    public:

        virtual ~IStatisticsSubset() { };

        /**
         * Marks the statistics at a specific index as missing, i.e., no condition that will be considered in the
         * following for refining a rule will be able to cover it and consequently the function `addToSubset` will never
         * be called for the given `statisticIndex`.
         *
         * @param statisticIndex    The index of the missing statistic
         * @param weight            The weight of the missing statistic
         */
        virtual void addToMissing(uint32 statisticIndex, float64 weight) = 0;

        /**
         * Adds the statistics at a specific index to the subset in order to mark it as covered by the condition that is
         * currently considered for refining a rule.
         *
         * This function must be called repeatedly for each statistic that is covered by the current condition,
         * immediately after the invocation of the function `Statistics#createSubset`. Each of these statistics must
         * have been provided earlier via the function `Statistics#addSampledStatistic` or
         * `Statistics#updateCoveredStatistic`.
         *
         * This function is supposed to update any internal state of the subset that relates to the statistics that are
         * covered by the current condition, i.e., to compute and store local information that is required by the other
         * functions that will be called later. Any information computed by this function is expected to be reset when
         * invoking the function `resetSubset` for the next time.
         *
         * @param statisticIndex    The index of the covered statistic
         * @param weight            The weight of the covered statistic
         */
        virtual void addToSubset(uint32 statisticIndex, float64 weight) = 0;

        /**
         * Resets the subset by removing all statistics that have been added via preceding calls to the function
         * `addToSubset`.
         *
         * This function is supposed to reset the internal state of the subset to the state when the subset was created
         * via the function `Statistics#createSubset`. When calling this function, the current state must not be purged
         * entirely, but it must be cached and made available for use by the functions `calculateExampleWisePrediction`
         * and `calculateLabelWisePrediction` (if the function argument `accumulated` is set accordingly).
         *
         * This function may be invoked multiple times with one or several calls to `addToSubset` in between, which is
         * supposed to update the previously cached state by accumulating the current one, i.e., the accumulated cached
         * state should be the same as if `resetSubset` would not have been called at all.
         */
        virtual void resetSubset() = 0;

        /**
         * Calculates and returns the scores to be predicted by a rule that covers all statistics that have been added
         * to the subset so far via the function `addToSubset`, as well as an overall quality score that assesses the
         * quality of the predicted scores.
         *
         * If the argument `uncovered` is true, the rule is considered to cover all statistics that belong to the
         * difference between the statistics that have been provided via the function `Statistics#addSampledStatistic`
         * or `Statistics#updateCoveredStatistic` and the statistics that have been added to the subset via the function
         * `addToSubset`.
         *
         * If the argument `accumulated` is true, all statistics that have been added since the subset has been created
         * via the function `Statistics#createSubset` are taken into account even if the function `resetSubset` has been
         * called since then. If said function has not been invoked, this argument does not have any effect.
         *
         * @param uncovered     False, if the rule covers all statistics that have been added to the subset via the
         *                      function `addToSubset`, true, if the rule covers all statistics that belong to the
         *                      difference between the statistics that have been provided via the function
         *                      `Statistics#addSampledStatistic` or `Statistics#updateCoveredStatistic` and the
         *                      statistics that have been added via the function `addToSubset`
         * @param accumulated   False, if the rule covers all statistics that have been added to the subset via the
         *                      function `addToSubset` since the function `resetSubset` has been called for the last
         *                      time, true, if the rule covers all examples that have been provided since the subset has
         *                      been created via the function `Statistics#createSubset`
         * @return              A reference to an object of type `IScoreVector` that stores the scores to be predicted
         *                      by the rule for each considered label, as well as an overall quality score
         */
        virtual const IScoreVector& calculatePrediction(bool uncovered, bool accumulated) = 0;

};
