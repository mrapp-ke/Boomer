from boomer.algorithm._arrays cimport float64


cdef class Shrinkage:

    # Functions:

    cdef apply_shrinkage(self, float64[::1] predicted_scores)


cdef class ConstantShrinkage(Shrinkage):

    # Attributes:

    cdef float shrinkage

    # Functions:

    cdef apply_shrinkage(self, float64[::1] predicted_scores)
