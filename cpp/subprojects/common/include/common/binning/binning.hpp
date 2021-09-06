/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <cmath>


/**
 * Calculates and returns the number of bins to be used, based on the number of values available, a percentage that
 * specifies how many bins should be used, and a minimum and maximum number of bins.
 *
 * @param numValues The number of values available
 * @param binRatio  A percentage that specifies how many bins should be used
 * @param minBins   The minimum number of bins
 * @param maxBins   The maximum number of bins or a value < `minBins`, if the maximum number should not be restricted
 */
static inline constexpr uint32 calculateNumBins(uint32 numValues, float32 binRatio, uint32 minBins, uint32 maxBins) {
    // Calculate number of bins based on the given percentage...
    uint32 numBins = std::ceil(binRatio * numValues);

    // Prevent the minimum number of bins to exceed the number of available values...
    uint32 min = minBins > numValues ? numValues : minBins;

    // Ensure that the number of bins is not smaller than the given minimum...
    if (numBins < min) {
        return min;
    }

    // If `maxBins >= minBins`, ensure that the number of bins does not exceed the given maximum...
    if (maxBins >= minBins && numBins > maxBins) {
        return maxBins;
    }

    return numBins;
}
