/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics_immutable.hpp"


/**
 * Defines an interface for all classes that provide access to statistics that are organized as a histogram, i.e., where
 * the statistics of multiple training examples are aggregated into the same bin.
 */
class IHistogram : virtual public IImmutableStatistics {

    public:

        virtual ~IHistogram() { };

        /**
         * Sets all statistics in the histogram to zero.
         */
        virtual void setAllToZero() = 0;

        /**
         * Adds the statistic at a specific index to a specific bin.
         *
         * @param binIndex          The index of the bin
         * @param statisticIndex    The index of the statistic
         * @param weight            The weight of the statistic
         */
        virtual void addToBin(uint32 binIndex, uint32 statisticIndex, uint32 weight) = 0;

};
