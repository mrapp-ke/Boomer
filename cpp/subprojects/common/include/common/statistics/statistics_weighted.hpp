/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/binning/bin_index_vector_dense.hpp"
#include "common/binning/bin_index_vector_dok.hpp"
#include "common/statistics/histogram.hpp"
#include "common/statistics/statistics_weighted_immutable.hpp"

/**
 * Defines an interface for all classes that inherit from `IImmutableWeightedStatistics`, but do also provide functions
 * that allow to only use a sub-sample of the available statistics.
 */
class IWeightedStatistics : virtual public IImmutableWeightedStatistics {
    public:

        virtual ~IWeightedStatistics() override {};

        /**
         * Creates and returns a copy of this object.
         *
         * @return An unique pointer to an object of type `IWeightedStatistics` that has been created
         */
        virtual std::unique_ptr<IWeightedStatistics> copy() const = 0;

        /**
         * Resets the statistics which should be considered in the following for refining an existing rule. The indices
         * of the respective statistics must be provided via subsequent calls to the function `addCoveredStatistic`.
         *
         * This function must be invoked each time an existing rule has been refined, i.e., when a new condition has
         * been added to its body, because this results in a subset of the statistics being covered by the refined rule.
         *
         * This function is supposed to reset any non-global internal state that only holds for a certain subset of the
         * available statistics and therefore becomes invalid when a different subset of the statistics should be used.
         */
        virtual void resetCoveredStatistics() = 0;

        /**
         * Adds a specific statistic to the subset that is covered by an existing rule and therefore should be
         * considered in the following for refining an existing rule.
         *
         * This function must be called repeatedly for each statistic that is covered by the existing rule, immediately
         * after the invocation of the function `resetCoveredStatistics`.
         *
         * This function is supposed to update any internal state that relates to the considered statistics, i.e., to
         * compute and store local information that is required by the other functions that will be called later. Any
         * information computed by this function is expected to be reset when invoking the function
         * `resetCoveredStatistics` for the next time.
         *
         * @param statisticIndex The index of the statistic that should be added
         */
        virtual void addCoveredStatistic(uint32 statisticIndex) = 0;

        /**
         * Removes a specific statistic from the subset that is covered by an existing rule and therefore should not be
         * considered in the following for refining an existing rule.
         *
         * This function must be called repeatedly for each statistic that is not covered anymore by the existing rule.
         *
         * This function is supposed to update any internal state that relates to the considered statistics, i.e., to
         * compute and store local information that is required by the other functions that will be called later. Any
         * information computed by this function is expected to be reset when invoking the function
         * `resetCoveredStatistics` for the next time.
         *
         * @param statisticIndex The index of the statistic that should be removed
         */
        virtual void removeCoveredStatistic(uint32 statisticIndex) = 0;

        /**
         * Creates and returns a new histogram based on the statistics.
         *
         * @param binIndexVector    A reference to an object of type `DenseBinIndexVector` that stores the indices of
         *                          the bins, individual examples have been assigned to
         * @param numBins           The number of bins in the histogram
         * @return                  An unique pointer to an object of type `IHistogram` that has been created
         */
        virtual std::unique_ptr<IHistogram> createHistogram(const DenseBinIndexVector& binIndexVector,
                                                            uint32 numBins) const = 0;

        /**
         * Creates and returns a new histogram based on the statistics.
         *
         * @param binIndexVector    A reference to an object of type `DokBinIndexVector` that stores the indices of the
         *                          bins, individual examples have been assigned to
         * @param numBins           The number of bins in the histogram
         * @return                  An unique pointer to an object of type `IHistogram` that has been created
         */
        virtual std::unique_ptr<IHistogram> createHistogram(const DokBinIndexVector& binIndexVector,
                                                            uint32 numBins) const = 0;
};
