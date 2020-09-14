"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to generate random numbers.
"""
DEF MAX_RANDOM = 0x7FFFFFFF


cdef class RNG:
    """
    Implements a fast random number generator using 32 bit XOR shifts (for details, see
    http://www.jstatsoft.org/v08/i14/paper).
    """

    def __cinit__(self, uint32 random_state):
        """
        :param random_state: The seed to be used by the random number generator
        """
        self.random_state = random_state

    cdef uint32 random(self, uint32 min, uint32 max):
        """
        Generates and returns a random number in [min, max).

        :param min: The minimum number (inclusive)
        :param max: The maximum number (exclusive)
        :return:    The random number that has been generated
        """
        cdef uint32* random_state = &self.random_state

        if random_state[0] == 0:
            random_state[0] = 1

        random_state[0] ^= <uint32>(random_state[0] << 13)
        random_state[0] ^= <uint32>(random_state[0] >> 17)
        random_state[0] ^= <uint32>(random_state[0] << 5)

        cdef uint32 random_number = random_state[0] % <uint32>(MAX_RANDOM + 1)
        return min + (random_number % (max - min))
