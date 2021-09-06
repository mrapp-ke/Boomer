#include "common/sampling/random.hpp"


const uint32 MAX_RANDOM = 0x7FFFFFFF;

RNG::RNG(uint32 randomState)
    : randomState_(randomState) {

}

uint32 RNG::random(uint32 min, uint32 max) {
    uint32* randomState = &randomState_;

    if (randomState[0] == 0) {
        randomState[0] = 1;
    }

    randomState[0] ^= (uint32) (randomState[0] << 13);
    randomState[0] ^= (uint32) (randomState[0] >> 17);
    randomState[0] ^= (uint32) (randomState[0] << 5);

    uint32 randomNumber = randomState[0] % (MAX_RANDOM + 1);
    return min + (randomNumber % (max - min));
}
