/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * Implements a fast random number generator using 32 bit XOR shifts (for details, see
 * http://www.jstatsoft.org/v08/i14/paper).
 */
class RNG final {

    private:

        uint32 randomState_;

    public:

        /**
         * @param randomState The seed to be used by the random number generator
         */
        RNG(uint32 randomState);

        /**
         * Generates and returns a random number in [min, max).
         *
         * @param min   The minimum number (inclusive)
         * @param max   The maximum number (exclusive)
         * @return      The random number that has been generated
         */
        uint32 random(uint32 min, uint32 max);

};
