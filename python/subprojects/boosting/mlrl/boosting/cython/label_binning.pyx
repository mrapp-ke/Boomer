from mlrl.common.cython._types cimport uint32, float32

from libcpp.memory cimport make_unique


cdef class LabelBinningFactory:
    """
    A wrapper for the pure virtual C++ class `ILabelBinningFactory`.
    """
    pass


cdef class EqualWidthLabelBinningFactory(LabelBinningFactory):
    """
    A wrapper for the C++ class `EqualWidthLabelBinningFactory`.
    """

    def __cinit__(self, float32 bin_ratio, uint32 min_bins, uint32 max_bins):
        """
        :param bin_ratio:   A percentage that specifies how many bins should be used to assign labels to, e.g., if 100
                            labels are available, 0.5 means that `ceil(0.5 * 100) = 50` bins should be used
        :param min_bins:    The minimum number of bins to be used to assign labels to. Must be at least 2
        :param max_bins:    The maximum number of bins to be used to assign labels to. Must be at least `minBins` or 0,
                            if the maximum number of bins should not be restricted
        """
        self.label_binning_factory_ptr = <unique_ptr[ILabelBinningFactory]>make_unique[EqualWidthLabelBinningFactoryImpl](
            bin_ratio, min_bins, max_bins)
