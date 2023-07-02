/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_evaluation/score_vector.hpp"

/**
 * Defines an interface for all classes that provide access to a subset of the statistics that are stored by an instance
 * of the class `IStatistics` and allows to calculate the scores to be predicted by rules that cover such a subset.
 */
class IStatisticsSubset {
    public:

        virtual ~IStatisticsSubset() {};

        /**
         * Returns whether the statistics at a specific index have a non-zero weight or not.
         *
         * @return True, if the statistics at the given index have a non-zero weight, false otherwise
         */
        virtual bool hasNonZeroWeight(uint32 statisticIndex) const = 0;

        /**
         * Adds the statistics at a specific index to the subset in order to mark it as covered by the condition that is
         * currently considered for refining a rule.
         *
         * This function must be called repeatedly for each statistic that is covered by the current condition,
         * immediately after the invocation of the function `IImmutableWeightedStatistics::createSubset`. If a rule has
         * already been refined, each of these statistics must have been marked as covered earlier via the function
         * `IWeightedStatistics::addCoveredStatistic` and must not have been marked as uncovered via the function
         * `IWeightedStatistics::removeCoveredStatistic`.
         *
         * This function is supposed to update any internal state of the subset that relates to the statistics that are
         * covered by the current condition, i.e., to compute and store local information that is required by the other
         * functions that will be called later. Any information computed by this function is expected to be reset when
         * invoking the function `resetSubset` for the next time.
         *
         * @param statisticIndex The index of the covered statistic
         */
        virtual void addToSubset(uint32 statisticIndex) = 0;

        /**
         * Calculates and returns the scores to be predicted by a rule that covers all statistics that have been added
         * to the subset via the function `addToSubset`, as well as a numerical score that assesses the overall quality
         * of the predicted scores.
         *
         * @return A reference to an object of type `IScoreVector` that stores the scores to be predicted by the rule
         *         for each considered label, as well as a numerical score that assesses their overall quality
         */
        virtual const IScoreVector& calculateScores() = 0;
};
