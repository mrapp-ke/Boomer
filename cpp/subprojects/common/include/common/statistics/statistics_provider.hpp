/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/statistics/statistics.hpp"


/**
 * Provides access to an object of type `IStatistics`.
 */
class IStatisticsProvider {

    public:

        virtual ~IStatisticsProvider() { };

        /**
         * Returns an object of type `IStatistics`.
         *
         * @return A reference to an object of type `IStatistics`
         */
        virtual IStatistics& get() const = 0;

        /**
         * Switches the implementation that is used for calculating the predictions of rules, as well as corresponding
         * quality scores, to the one that should be used for learning regular rules.
         */
        virtual void switchToRegularRuleEvaluation() = 0;

        /**
         * Switches the implementation that is used for calculating the predictions of rules, as well as corresponding
         * quality scores, to the one that should be used for pruning rules.
         */
        virtual void switchToPruningRuleEvaluation() = 0;

};
