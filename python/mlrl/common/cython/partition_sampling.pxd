from mlrl.common.cython._types cimport float32

from libcpp.memory cimport unique_ptr


cdef extern from "common/sampling/partition_sampling.hpp" nogil:

    cdef cppclass IPartitionSamplingFactory:
        pass


cdef extern from "common/sampling/partition_sampling_no.hpp" nogil:

    cdef cppclass NoPartitionSamplingFactoryImpl"NoPartitionSamplingFactory"(IPartitionSamplingFactory):
        pass


cdef extern from "common/sampling/partition_sampling_bi_random.hpp" nogil:

    cdef cppclass RandomBiPartitionSamplingFactoryImpl"RandomBiPartitionSamplingFactory"(IPartitionSamplingFactory):

        # Constructors:

        RandomBiPartitionSamplingFactoryImpl(float32 holdout_set_size) except +


cdef extern from "common/sampling/partition_sampling_bi_stratified_example_wise.hpp" nogil:

    cdef cppclass ExampleWiseStratifiedBiPartitionSamplingFactoryImpl"ExampleWiseStratifiedBiPartitionSamplingFactory"(
            IPartitionSamplingFactory):

        # Constructors:

        ExampleWiseStratifiedBiPartitionSamplingFactoryImpl(float32 holdout_set_size) except +


cdef extern from "common/sampling/partition_sampling_bi_stratified_label_wise.hpp" nogil:

    cdef cppclass LabelWiseStratifiedBiPartitionSamplingFactoryImpl"LabelWiseStratifiedBiPartitionSamplingFactory"(
            IPartitionSamplingFactory):

        # Constructors:

        LabelWiseStratifiedBiPartitionSamplingFactoryImpl(float32 holdout_set_size) except +


cdef class PartitionSamplingFactory:

    # Attributes:

    cdef unique_ptr[IPartitionSamplingFactory] partition_sampling_factory_ptr


cdef class NoPartitionSamplingFactory(PartitionSamplingFactory):
    pass


cdef class RandomBiPartitionSamplingFactory(PartitionSamplingFactory):
    pass


cdef class ExampleWiseStratifiedBiPartitionSamplingFactory(PartitionSamplingFactory):
    pass


cdef class LabelWiseStratifiedBiPartitionSamplingFactory(PartitionSamplingFactory):
    pass
