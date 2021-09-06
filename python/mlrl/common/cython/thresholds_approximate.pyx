"""
@author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.common.cython.feature_binning cimport FeatureBinning

from libcpp.utility cimport move
from libcpp.memory cimport unique_ptr, make_unique


cdef class ApproximateThresholdsFactory(ThresholdsFactory):
    """
    A wrapper for the C++ class `ApproximateThresholdsFactory`.
    """

    def __cinit__(self, FeatureBinning binning not None, uint32 num_threads):
        """
        :param binning:     The binning method to be used
        :param num_threads: The number of CPU threads to be used to update statistics in parallel. Must be at least 1
        """
        self.thresholds_factory_ptr = <unique_ptr[IThresholdsFactory]>make_unique[ApproximateThresholdsFactoryImpl](
            move(binning.binning_ptr), num_threads)
