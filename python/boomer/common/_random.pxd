from boomer.common._arrays cimport uint32


cdef class RNG:

    # Attributes:

    cdef readonly uint32 random_state

    # Functions:

    cdef uint32 random(self, uint32 min, uint32 max)
