"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._validation import assert_greater, assert_less


cdef class ExampleWiseStratifiedBiPartitionSamplingConfig:
    """
    Allows to configure a method for partitioning the available training examples into a training set and a holdout set
    using stratification, where distinct label vectors are treated as individual classes.
    """

    def get_holdout_set_size(self) -> float:
        """
        Returns the fraction of examples that are included in the holdout set.

        :return: The fraction of examples that are included in the holdout set
        """
        return self.config_ptr.getHoldoutSetSize()

    def set_holdout_set_size(self, holdout_set_size: float) -> ExampleWiseStratifiedBiPartitionSamplingConfig:
        """
        Sets the fraction of examples that should be included in the holdout set.

        :param holdout_set_size:    The fraction of examples that should be included in the holdout set, e.g., a value
                                    of 0.6 corresponds to 60 % of the available examples. Must be in (0, 1)
        :return:                    An `ExampleWiseStratifiedBiPartitionSamplingConfig` that allows further
                                    configuration of the method for partitioning the available training examples into
                                    a training set and a holdout set
        """
        assert_greater('holdout_set_size', holdout_set_size, 0)
        assert_less('holdout_set_size', holdout_set_size, 1)
        self.config_ptr.setHoldoutSetSize(holdout_set_size)
        return self


cdef class LabelWiseStratifiedBiPartitionSamplingConfig:
    """
    Allows to configure a method for partitioning the available training examples into a training set and a holdout set
    using stratification, such that for each label the proportion of relevant and irrelevant examples is maintained.
    """

    def get_holdout_set_size(self) -> float:
        """
        Returns the fraction of examples that are included in the holdout set.

        :return: The fraction of examples that are included in the holdout set
        """
        return self.config_ptr.getHoldoutSetSize()

    def set_holdout_set_size(self, holdout_set_size: float) -> LabelWiseStratifiedBiPartitionSamplingConfig:
        """
        Sets the fraction of examples that should be included in the holdout set.

        :param holdout_set_size:    The fraction of examples that should be included in the holdout set, e.g., a value
                                    of 0.6 corresponds to 60 % of the available examples. Must be in (0, 1)
        :return:                    An `LabelWiseStratifiedBiPartitionSamplingConfig` that allows further configuration
                                    of the method for partitioning the available training examples into a training set
                                    and a holdout set
        """
        assert_greater('holdout_set_size', holdout_set_size, 0)
        assert_less('holdout_set_size', holdout_set_size, 1)
        self.config_ptr.setHoldoutSetSize(holdout_set_size)


cdef class RandomBiPartitionSamplingConfig:
    """
    Allows to configure a method for partitioning the available training examples into a training set and a holdout set
    that randomly splits the training examples into two mutually exclusive sets.
    """

    def get_holdout_set_size(self) -> float:
        """
        Returns the fraction of examples that are included in the holdout set.

        :return: The fraction of examples that are included in the holdout set
        """
        return self.config_ptr.getHoldoutSetSize()

    def set_holdout_set_size(self, holdout_set_size: float) -> RandomBiPartitionSamplingConfig:
        """
        Sets the fraction of examples that should be included in the holdout set.

        :param holdout_set_size:    The fraction of examples that should be included in the holdout set, e.g., a value
                                    of 0.6 corresponds to 60 % of the available examples. Must be in (0, 1)
        :return:                    A `RandomBiPartitionSamplingConfig` that allows further configuration of the method
                                    for partitioning the available training examples into a training set and a holdout
                                    set
        """
        assert_greater('holdout_set_size', holdout_set_size, 0)
        assert_less('holdout_set_size', holdout_set_size, 1)
        self.config_ptr.setHoldoutSetSize(holdout_set_size)
