/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/weight_vector_dense.hpp"
#include <unordered_set>


/**
 * Randomly selects `numSamples` out of `numTotal` elements and sets their weights to 1, while the remaining weights are
 * set to 0, by using a set to keep track of the elements that have already been selected. This method is suitable if
 * `numSamples` is much smaller than `numTotal`.
 *
 * @tparam T            The type of the iterator that provides random access to the indices of the available elements to
 *                      sample from
 * @param iterator      An iterator that provides random access to the indices of the available elements to sample from
 * @param numTotal      The total number of available elements to sample from
 * @param numSamples    The number of weights to be set to 1
 * @param numElements   The total number of elements, including those that are not accessible via the given iterator
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 * @return              An unique pointer to an object of type `IWeightVector` that provides access to the weights
 */
template<class T>
static inline std::unique_ptr<IWeightVector> sampleWeightsWithoutReplacementViaTrackingSelection(T iterator,
                                                                                                 uint32 numTotal,
                                                                                                 uint32 numSamples,
                                                                                                 uint32 numElements,
                                                                                                 RNG& rng) {
    std::unique_ptr<DenseWeightVector> weightVectorPtr = std::make_unique<DenseWeightVector>(numElements, numSamples);
    DenseWeightVector::iterator sampleIterator = weightVectorPtr->begin();
    std::unordered_set<uint32> selectedIndices;

    for (uint32 i = 0; i < numSamples; i++) {
        bool shouldContinue = true;
        uint32 sampledIndex;

        while (shouldContinue) {
            uint32 randomIndex = rng.random(0, numTotal);
            sampledIndex = iterator[randomIndex];
            shouldContinue = !selectedIndices.insert(sampledIndex).second;
        }

        sampleIterator[sampledIndex] = 1;
    }

    return weightVectorPtr;
}

/**
 * Randomly selects `numSamples` out of `numTotal` elements and sets their weights to 1, while the remaining weights are
 * set to 0, by using a pool, i.e., an array, to keep track of the elements that have not been selected yet.
 *
 * @tparam T            The type of the iterator that provides random access to the indices of the available elements to
 *                      sample from
 * @param iterator      An iterator that provides random access to the indices of the available elements to sample from
 * @param numTotal      The total number of available elements to sample from
 * @param numSamples    The number of weights to be set to 1
 * @param numElements   The total number of elements, including those that are not accessible via the given iterator
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 * @return              An unique pointer to an object of type `IWeightVector` that provides access to the weights
 */
template<class T>
static inline std::unique_ptr<IWeightVector> sampleWeightsWithoutReplacementViaPool(T iterator, uint32 numTotal,
                                                                                    uint32 numSamples,
                                                                                    uint32 numElements, RNG& rng) {
    std::unique_ptr<DenseWeightVector> weightVectorPtr = std::make_unique<DenseWeightVector>(numElements, numSamples);
    DenseWeightVector::iterator sampleIterator = weightVectorPtr->begin();
    uint32 pool[numTotal];

    // Initialize pool...
    for (uint32 i = 0; i < numTotal; i++) {
        pool[i] = iterator[i];
    }

    for (uint32 i = 0; i < numSamples; i++) {
        // Randomly select an index that has not been drawn yet...
        uint32 randomIndex = rng.random(0, numTotal - i);
        uint32 sampledIndex = pool[randomIndex];

        // Set weight at the selected index to 1...
        sampleIterator[sampledIndex] = 1;

        // Move the index at the border to the position of the recently drawn index...
        pool[randomIndex] = pool[numTotal - i - 1];
    }

    return weightVectorPtr;
}

/**
 * Randomly selects `numSamples` out of `numTotal` elements and sets their weights to 1, while the remaining weights are
 * set to 0. The method that is used internally is chosen automatically, depending on the ratio `numSamples / numTotal`.
 *
 * @tparam T            The type of the iterator that provides random access to the indices of the available elements to
 *                      sample from
 * @param iterator      An iterator that provides random access to the indices of the available elements to sample from
 * @param numTotal      The total number of available elements to sample from
 * @param numSamples    The number of weights to be set to 1
 * @param numElements   The total number of elements, including those that are not accessible via the given iterator
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 * @return              An unique pointer to an object of type `IWeightVector` that provides access to the weights
 *
 */
template<class T>
static inline std::unique_ptr<IWeightVector> sampleWeightsWithoutReplacement(T iterator, uint32 numTotal,
                                                                             uint32 numSamples, uint32 numElements,
                                                                             RNG& rng) {
    float64 ratio = numTotal > 0 ? ((float64) numSamples) / ((float64) numTotal) : 1;

    if (ratio < 0.06) {
        // For very small ratios use tracking selection
        return sampleWeightsWithoutReplacementViaTrackingSelection(iterator, numTotal, numSamples, numElements, rng);
    } else {
        // Otherwise, use a pool as the default method
        return sampleWeightsWithoutReplacementViaPool(iterator, numTotal, numSamples, numElements, rng);
    }
}
