"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport unique_ptr, make_unique


cdef class ExactThresholdsFactory(ThresholdsFactory):
    """
    A wrapper for the C++ class `ExactThresholdsFactory`.
    """

    def __cinit__(self, uint32 num_threads):
        """
        :param num_threads: The number of CPU threads to be used to update statistics in parallel. Must be at least 1
        """
        self.thresholds_factory_ptr = <unique_ptr[IThresholdsFactory]>make_unique[ExactThresholdsFactoryImpl](
            num_threads)
