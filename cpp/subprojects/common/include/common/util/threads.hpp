/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include <algorithm>
#include <thread>


/**
 * Returns the number of threads that are available for parallelized algorithms.
 *
 * @param numPreferredThreads   The preferred number of threads or 0, if all available CPU cores should be used
 * @return                      The number of available threads
 */
static inline uint32 getNumAvailableThreads(uint32 numPreferredThreads) {
    uint32 numAvailableThreads = std::max<uint32>(std::thread::hardware_concurrency(), 1);
    return numPreferredThreads > 0 ? std::min(numAvailableThreads, numPreferredThreads) : numAvailableThreads;
}
