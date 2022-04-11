"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._validation import assert_greater_or_equal


cdef class LabelSamplingWithoutReplacementConfig:
    """
    Allows to configure a method for sampling labels without replacement.
    """

    def get_num_samples(self) -> int:
        """
        Returns the number of labels that are included in a sample.

        :return: The number of labels that are included in a sample
        """
        return self.config_ptr.getNumSamples()

    def set_num_samples(self, num_samples: int) -> LabelSamplingWithoutReplacementConfig:
        """
        Sets the number of labels that should be included in a sample.

        :param num_samples: The number of labels that should be included in a sample. Must be at least 1
        :return:            A `LabelSamplingWithoutReplacementConfig` that allows further configuration of the method
                            for sampling labels
        """
        assert_greater_or_equal('num_samples', num_samples, 1)
        self.config_ptr.setNumSamples(num_samples)
        return self
