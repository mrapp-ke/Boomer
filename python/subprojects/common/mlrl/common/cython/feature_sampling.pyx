"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater_or_equal, assert_less


cdef class FeatureSamplingWithoutReplacementConfig:
    """
    Allows to configure a method for sampling features without replacement.
    """

    def get_sample_size(self) -> float:
        """
        Returns the fraction of features that are included in a sample.

        :return: The fraction of features that are included in a sample
        """
        return self.config_ptr.getSampleSize()

    def set_sample_size(self, sample_size: float) -> FeatureSamplingWithoutReplacementConfig:
        """
        Sets the fraction of features that should be included in a sample.

        :param sample_size: The fraction of features that should be included in a sample, e.g., a value of 0.6
                            corresponds to 60 % of the available features. Must be in (0, 1) or 0, if the default sample
                            size `floor(log2(numFeatures - 1) + 1)` should be used
        :return:            A `FeatureSamplingWithoutReplacementConfig` that allows further configuration of the method
                            for sampling features
        """
        assert_greater_or_equal('sample_size', sample_size, 0)
        assert_less('sample_size', sample_size, 1)
        self.config_ptr.setSampleSize(sample_size)
        return self

    def get_num_retained(self) -> int:
        """
        Returns the number of trailing features that are always included in a sample.
        
        :return: The number of trailing features that are always included in a sample
        """
        return self.config_ptr.getNumRetained()
    
    def set_num_retained(self, num_retained: int) -> FeatureSamplingWithoutReplacementConfig:
        """
        Sets the number fo trailing features that should always be included in a sample.

        :param num_retained:    The number of trailing features that should always be included in a sample. Must be at
                                least 0
        :return:                A `FeatureSamplingWithoutReplacementConfig` that allows further configuration of the
                                method for sampling features
        """
        assert_greater_or_equal('num_retained', num_retained, 0)
        self.config_ptr.setNumRetained(num_retained)
        return self
