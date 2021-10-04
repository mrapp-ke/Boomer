from mlrl.common.cython._types cimport uint32
from mlrl.common.cython.thresholds cimport ThresholdsFactory, IThresholdsFactory


cdef extern from "common/thresholds/thresholds_exact.hpp" nogil:

    cdef cppclass ExactThresholdsFactoryImpl"ExactThresholdsFactory"(IThresholdsFactory):

        # Constructors:

        ExactThresholdsFactoryImpl(uint32 numThreads) except +


cdef class ExactThresholdsFactory(ThresholdsFactory):
    pass
