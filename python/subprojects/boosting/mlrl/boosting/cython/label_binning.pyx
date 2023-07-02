"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater, assert_greater_or_equal, assert_less


cdef class EqualWidthLabelBinningConfig:
    """
    Allows to configure a method that assigns labels to bins in a way such that each bin contains labels for which the
    predicted score is expected to belong to the same value range.
    """

    def get_bin_ratio(self) -> float:
        """
        Returns the percentage that specifies how many bins are used.

        :return: The percentage that specifies how many bins are used
        """
        return self.config_ptr.getBinRatio()

    def set_bin_ratio(self, bin_ratio: float) -> EqualWidthLabelBinningConfig:
        """
        Sets the percentage that specifies how many should be used.

        :param bin_ratio:   A percentage that specifies how many bins should be used, e.g., if 100 labels are
                            available, a percentage of 0.5 means that `ceil(0.5 * 100) = 50` bins should be used. Must
                            be in (0, 1)
        :return:            An `EqualWidthLabelBinningConfig` that allows further configuration of the method that
                            assigns labels to bins
        """
        assert_greater('bin_ratio', bin_ratio, 0)
        assert_less('bin_ratio', bin_ratio, 1)
        self.config_ptr.setBinRatio(bin_ratio)
        return self

    def get_min_bins(self) -> int:
        """
        Returns the minimum number of bins that is used.

        :return: The minimum number of bins that is used
        """
        return self.config_ptr.getMinBins()

    def set_min_bins(self, min_bins: int) -> EqualWidthLabelBinningConfig:
        """
        Sets the minimum number of bins that should be used.

        :param min_bins:    The minimum number of bins that should be used. Must be at least 1
        :return:            An `EqualWidthLabelBinningConfig` that allows further configuration of the method that
                            assigns labels to bins
        """
        assert_greater_or_equal('min_bins', min_bins, 1)
        self.config_ptr.setMinBins(min_bins)
        return self

    def get_max_bins(self) -> int:
        """
        Returns the maximum number of bins that is used.

        :return: The maximum number of bins that is used
        """
        return self.config_ptr.getMaxBins()

    def set_max_bins(self, max_bins: int) -> EqualWidthLabelBinningConfig:
        """
        Sets the maximum number of bins that should be used.

        :param max_bins:    The maximum number of bins that should be used. Must be at least the minimum number of bins
                            or 0, if the maximum number of bins should not be restricted
        :return:            An `EqualWidthLabelBinningConfig` that allows further configuration of the method that
                            assigns labels to bins
        """
        if max_bins != 0:
            assert_greater_or_equal('max_bins', max_bins, self.config_ptr.getMinBins())
        self.config_ptr.setMaxBins(max_bins)
        return self
