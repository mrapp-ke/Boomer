from boomer.common._arrays cimport float64
from boomer.common._predictions cimport Prediction
from boomer.common.post_processing cimport PostProcessor


cdef class Shrinkage(PostProcessor):

    # Functions:

    cdef void post_process(self, Prediction* prediction)


cdef class ConstantShrinkage(Shrinkage):

    # Attributes:

    cdef float64 shrinkage

    # Functions:

    cdef void post_process(self, Prediction* prediction)
