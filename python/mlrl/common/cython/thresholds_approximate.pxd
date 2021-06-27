from mlrl.common.cython._types cimport uint32
from mlrl.common.cython.binning cimport IFeatureBinning
from mlrl.common.cython.thresholds cimport ThresholdsFactory, IThresholdsFactory

from libcpp.memory cimport shared_ptr


cdef extern from "common/thresholds/thresholds_approximate.hpp" nogil:

    cdef cppclass ApproximateThresholdsFactoryImpl"ApproximateThresholdsFactory"(IThresholdsFactory):

        # Constructors:

        ApproximateThresholdsFactory(shared_ptr[IFeatureBinning] binningPtr, uint32 numThreads) except +


cdef class ApproximateThresholdsFactory(ThresholdsFactory):
    pass
