/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/statistics/statistics_weighted_immutable.hpp"

/**
 * Defines an interface for all classes that provide access to statistics that are organized as a histogram, i.e., where
 * the statistics of multiple training examples are aggregated into the same bin.
 */
class IHistogram : virtual public IImmutableWeightedStatistics {
    public:

        virtual ~IHistogram() override {};

        /**
         * Sets all statistics in the histogram to zero.
         */
        virtual void clear() = 0;

        /**
         * Returns the weight of the bin at a specific index, i.e., the number of statistics that have been assigned to
         * it.
         *
         * @param binIndex  The index of the bin
         * @return          The weight of the bin
         */
        virtual uint32 getBinWeight(uint32 binIndex) const = 0;

        /**
         * Adds the statistic at a specific index to the corresponding bin.
         *
         * @param statisticIndex The index of the statistic
         */
        virtual void addToBin(uint32 statisticIndex) = 0;
};
