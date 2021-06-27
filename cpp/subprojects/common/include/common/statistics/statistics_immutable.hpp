/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics_subset.hpp"
#include "common/indices/index_vector_full.hpp"
#include "common/indices/index_vector_partial.hpp"
#include <memory>


/**
 * Defines an interface for all classes that provide access to statistics about the labels of the training examples,
 * which serve as the basis for learning a new rule or refining an existing one.
 */
class IImmutableStatistics {

    public:

        virtual ~IImmutableStatistics() { };

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
         * Creates a new, empty subset of the statistics that includes only those labels, whose indices are provided by
         * a specific `FullIndexVector`. Individual statistics that are covered by a refinement of a rule can be added
         * to the subset via subsequent calls to the function `IStatisticsSubset#addToSubset`.
         *
         * This function, or the function `createSubset(PartialIndexVector&)` must be called each time a new refinement
         * is considered, unless the refinement covers all statistics previously provided via calls to the function
         * `IStatisticsSubset#addToSubset`.
         *
         * @param labelIndices  A reference to an object of type `FullIndexVector` that provides access to the indices
         *                      of the labels that should be included in the subset
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const FullIndexVector& labelIndices) const = 0;

        /**
         * Creates a new, empty subset of the statistics that includes only those labels, whose indices are provided by
         * a specific `PartialIndexVector`. Individual statistics that are covered by a refinement of a rule can be
         * added to the subset via subsequent calls to the function `IStatisticsSubset#addToSubset`.
         *
         * This function, or the function `createSubset(FullIndexVector&)` must be called each time a new refinement is
         * considered, unless the refinement covers all statistics previously provided via calls to the function
         * `IStatisticsSubset#addToSubset`.
         *
         * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices) const = 0;

};
