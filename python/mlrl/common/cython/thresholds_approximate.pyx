"""
@author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.common.cython.binning cimport FeatureBinning

from libcpp.memory cimport make_shared


cdef class ApproximateThresholdsFactory(ThresholdsFactory):
    """
    A wrapper for the C++ class `ApproximateThresholdsFactory`.
    """

    def __cinit__(self, FeatureBinning binning, uint32 num_threads):
        """
        :param binning:     The binning method to be used
        :param num_threads: The number of CPU threads to be used to update statistics in parallel. Must be at least 1
        """
        self.thresholds_factory_ptr = <shared_ptr[IThresholdsFactory]>make_shared[ApproximateThresholdsFactoryImpl](
            binning.binning_ptr, num_threads)
