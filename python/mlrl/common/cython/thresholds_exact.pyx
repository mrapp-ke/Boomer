"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport shared_ptr, make_shared


cdef class ExactThresholdsFactory(ThresholdsFactory):
    """
    A wrapper for the C++ class `ExactThresholdsFactory`.
    """

    def __cinit__(self, uint32 num_threads):
        """
        :param num_threads: The number of CPU threads to be used to update statistics in parallel. Must be at least 1
        """
        self.thresholds_factory_ptr = <shared_ptr[IThresholdsFactory]>make_shared[ExactThresholdsFactoryImpl](
            num_threads)
