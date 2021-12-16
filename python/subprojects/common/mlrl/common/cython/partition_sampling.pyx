"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport make_unique


cdef class PartitionSamplingFactory:
    """
    A wrapper for the pure virtual C++ class `IPartitionSamplingFactory`.
    """
    pass


cdef class NoPartitionSamplingFactory(PartitionSamplingFactory):
    """
    A wrapper for hte C++ class `NoPartitionSamplingFactory`.
    """

    def __cinit__(self):
        self.partition_sampling_factory_ptr = <unique_ptr[IPartitionSamplingFactory]>make_unique[NoPartitionSamplingFactoryImpl]()


cdef class RandomBiPartitionSamplingFactory(PartitionSamplingFactory):
    """
    A wrapper for the C++ class `RandomBiPartitionSamplingFactory`.
    """

    def __cinit__(self, float32 holdout_set_size):
        """
        :param holdout_set_size: The fraction of examples to be included in the holdout set (e.g. a value of 0.6
                                 corresponds to 60 % of the available examples). Must be in (0, 1)
        """
        self.partition_sampling_factory_ptr = <unique_ptr[IPartitionSamplingFactory]>make_unique[RandomBiPartitionSamplingFactoryImpl](
            holdout_set_size)


cdef class ExampleWiseStratifiedBiPartitionSamplingFactory(PartitionSamplingFactory):
    """
    A wrapper for the C++ class `ExampleWiseStratifiedBiPartitionSamplingFactory`.
    """

    def __cinit__(self, float32 holdout_set_size):
        """
        :param holdout_set_size: The fraction of examples to be included in the holdout set (e.g. a value of 0.6
                                 corresponds to 60 % of the available examples). Must be in (0, 1)
        """
        self.partition_sampling_factory_ptr = <unique_ptr[IPartitionSamplingFactory]>make_unique[ExampleWiseStratifiedBiPartitionSamplingFactoryImpl](
            holdout_set_size)


cdef class LabelWiseStratifiedBiPartitionSamplingFactory(PartitionSamplingFactory):
    """
    A wrapper for the C++ class `LabelWiseStratifiedBiPartitionSamplingFactory`.
    """

    def __cinit__(self, float32 holdout_set_size):
        """
        :param holdout_set_size: The fraction of examples to be included in the holdout set (e.g. a value of 0.6
                                 corresponds to 60 % of the available examples). Must be in (0, 1)
        """
        self.partition_sampling_factory_ptr = <unique_ptr[IPartitionSamplingFactory]>make_unique[LabelWiseStratifiedBiPartitionSamplingFactoryImpl](
            holdout_set_size)
