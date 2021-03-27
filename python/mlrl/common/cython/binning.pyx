"""
@author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_shared


cdef class FeatureBinning:
    """
    A wrapper for the pure virtual C++ class `IFeatureBinning`.
    """
    pass


cdef class EqualFrequencyFeatureBinning(FeatureBinning):
    """
    A wrapper for the C++ class `EqualFrequencyFeatureBinning`.
    """

    def __cinit__(self, float32 bin_ratio, uint32 min_bins, uint32 max_bins):
        """
        :param bin_ratio:   A percentage that specifies how many bins should be used
        :param min_bins:    The minimum number of bins to be used
        :param max_bins:    The maximum number of bins to be used
        """
        self.binning_ptr = <shared_ptr[IFeatureBinning]>make_shared[EqualFrequencyFeatureBinningImpl](bin_ratio,
                                                                                                      min_bins,
                                                                                                      max_bins)


cdef class EqualWidthFeatureBinning(FeatureBinning):
    """
    A wrapper for the C++ class `EqualWidthFeatureBinning`.
    """

    def __cinit__(self, float32 bin_ratio, uint32 min_bins, uint32 max_bins):
        """
        :param bin_ratio:   A percentage that specifies how many bins should be used
        :param min_bins:    The minimum number of bins to be used
        :param max_bins:    The maximum number of bins to be used
        """
        self.binning_ptr = <shared_ptr[IFeatureBinning]>make_shared[EqualWidthFeatureBinningImpl](bin_ratio, min_bins,
                                                                                                  max_bins)
