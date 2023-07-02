/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/statistics/statistics_subset_weighted.hpp"

#include <memory>

/**
 * Defines an interface for all classes that provide access to weighted statistics about the labels of the training
 * examples, which serve as the basis for learning a new rule or refining an existing one.
 */
class IImmutableWeightedStatistics {
    public:

        virtual ~IImmutableWeightedStatistics() {};

        /**
         * Returns the number of available statistics.
         *
         * @return The number of statistics
         */
        virtual uint32 getNumStatistics() const = 0;

        /**
         * Returns the number of available labels.
         *
         * @return The number of labels
         */
        virtual uint32 getNumLabels() const = 0;

        /**
         * Creates and returns a new object of type `IWeightedStatisticsSubset` that includes only those labels, whose
         * indices are provided by a specific `CompleteIndexVector`.
         *
         * @param labelIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @return              An unique pointer to an object of type `IWeightedStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IWeightedStatisticsSubset> createSubset(
          const CompleteIndexVector& labelIndices) const = 0;

        /**
         * Creates and returns a new object of type `IWeightedStatisticsSubset` that includes only those labels, whose
         * indices are provided by a specific `PartialIndexVector`.
         *
         * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @return              An unique pointer to an object of type `IWeightedStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IWeightedStatisticsSubset> createSubset(
          const PartialIndexVector& labelIndices) const = 0;
};
