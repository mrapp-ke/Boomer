/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/random.hpp"


static inline bool tiebreak(uint32 numDesiredSamples, uint32 numDesiredOutOfSamples, RNG& rng) {
    if (numDesiredSamples > numDesiredOutOfSamples) {
        return true;
    } else if (numDesiredSamples < numDesiredOutOfSamples) {
        return false;
    } else {
        return rng.random(0, 2) != 0;
    }
}
