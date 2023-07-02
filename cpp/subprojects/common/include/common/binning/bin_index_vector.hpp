/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"

#include <memory>

// Forward declarations
class IHistogram;
class IWeightedStatistics;

/**
 * Defines an interface for all classes that provide access to the indices of the bins, individual examples have been
 * assigned to.
 */
class IBinIndexVector {
    public:

        /**
         * The index of the bin that contains sparse values.
         */
        static const uint32 BIN_INDEX_SPARSE = std::numeric_limits<uint32>::max();

        virtual ~IBinIndexVector() {};

        /**
         * Returns the index of the bin, the example at a specific index has been assigned to.
         *
         * @param exampleIndex  The index of the example
         * @return              The index of the bin, the example has been assigned to
         */
        virtual uint32 getBinIndex(uint32 exampleIndex) const = 0;

        /**
         * Sets the index of the bin, the examples at a specific index should be assigned to.
         *
         * @param exampleIndex  The index of the example
         * @param binIndex      The index of the bin, the example should be assigned to
         */
        virtual void setBinIndex(uint32 exampleIndex, uint32 binIndex) = 0;

        /**
         * Creates and returns a new histogram based on given statistics and the indices that are stored by this vector.
         *
         * @param statistics    A reference to an object of type `IWeightedStatistics` that should be used
         * @param numBins       The number of bins in the histogram
         * @return              An unique pointer to an object of type `IHistogram` that has been created
         */
        virtual std::unique_ptr<IHistogram> createHistogram(const IWeightedStatistics& statistics,
                                                            uint32 numBins) const = 0;
};
