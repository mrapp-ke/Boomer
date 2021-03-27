/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
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
         * Allows to switch the implementation that is used for calculating the predictions of rules, as well as
         * corresponding quality scores, from the one that was initially used for learning the default rule, to another
         * one that will be used for all remaining rules.
         */
        virtual void switchRuleEvaluation() = 0;

};
