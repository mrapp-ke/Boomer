from mlrl.common.cython._types cimport uint32, float32

from libcpp.memory cimport shared_ptr


cdef extern from "common/sampling/random.hpp" nogil:

    cdef cppclass RNG:

        # Constructors:

        RNG(uint32 randomState) except +


cdef extern from "common/sampling/weight_vector.hpp" nogil:

    cdef cppclass IWeightVector:
        pass


cdef extern from "common/sampling/instance_sampling.hpp" nogil:

    cdef cppclass IInstanceSubSampling:
        pass


cdef extern from "common/sampling/instance_sampling_bagging.hpp" nogil:

    cdef cppclass BaggingImpl"Bagging"(IInstanceSubSampling):

        # Constructors:

        BaggingImpl(float32 sampleSize) except +


cdef extern from "common/sampling/instance_sampling_random.hpp" nogil:

    cdef cppclass RandomInstanceSubsetSelectionImpl"RandomInstanceSubsetSelection"(IInstanceSubSampling):

        # Constructors:

        RandomInstanceSubsetSelectionImpl(float32 sampleSize)


cdef extern from "common/sampling/instance_sampling_no.hpp" nogil:

    cdef cppclass NoInstanceSubSamplingImpl"NoInstanceSubSampling"(IInstanceSubSampling):
        pass


cdef extern from "common/sampling/feature_sampling.hpp" nogil:

    cdef cppclass IFeatureSubSampling:
        pass


cdef extern from "common/sampling/feature_sampling_random.hpp" nogil:

    cdef cppclass RandomFeatureSubsetSelectionImpl"RandomFeatureSubsetSelection"(IFeatureSubSampling):

        # Constructors:

        RandomFeatureSubsetSelectionImpl(float32 sampleSize) except +


cdef extern from "common/sampling/feature_sampling_no.hpp" nogil:

    cdef cppclass NoFeatureSubSamplingImpl"NoFeatureSubSampling"(IFeatureSubSampling):
        pass


cdef extern from "common/sampling/label_sampling.hpp" nogil:

    cdef cppclass ILabelSubSampling:
        pass


cdef extern from "common/sampling/label_sampling_random.hpp" nogil:

    cdef cppclass RandomLabelSubsetSelectionImpl"RandomLabelSubsetSelection"(ILabelSubSampling):

        # Constructors:

        RandomLabelSubsetSelectionImpl(uint32 numSamples)


cdef extern from "common/sampling/label_sampling_no.hpp" nogil:

    cdef cppclass NoLabelSubSamplingImpl"NoLabelSubSampling"(ILabelSubSampling):
        pass


cdef extern from "common/sampling/partition_sampling.hpp" nogil:

    cdef cppclass IPartitionSampling:
        pass


cdef extern from "common/sampling/partition_sampling_no.hpp" nogil:

    cdef cppclass NoPartitionSamplingImpl"NoPartitionSampling"(IPartitionSampling):
        pass


cdef extern from "common/sampling/partition_sampling_bi.hpp" nogil:

    cdef cppclass BiPartitionSamplingImpl"BiPartitionSampling"(IPartitionSampling):
        pass


cdef class InstanceSubSampling:

    # Attributes:

    cdef shared_ptr[IInstanceSubSampling] instance_sub_sampling_ptr


cdef class Bagging(InstanceSubSampling):
    pass


cdef class RandomInstanceSubsetSelection(InstanceSubSampling):
    pass


cdef class NoInstanceSubSampling(InstanceSubSampling):
    pass


cdef class FeatureSubSampling:

    # Attributes:

    cdef shared_ptr[IFeatureSubSampling] feature_sub_sampling_ptr


cdef class RandomFeatureSubsetSelection(FeatureSubSampling):
    pass


cdef class NoFeatureSubSampling(FeatureSubSampling):
    pass


cdef class LabelSubSampling:

    # Attributes:

    cdef shared_ptr[ILabelSubSampling] label_sub_sampling_ptr


cdef class RandomLabelSubsetSelection(LabelSubSampling):
    pass


cdef class NoLabelSubSampling(LabelSubSampling):
    pass


cdef class PartitionSampling:

    # Attributes:

    cdef shared_ptr[IPartitionSampling] partition_sampling_ptr


cdef class NoPartitionSampling(PartitionSampling):
    pass


cdef class BiPartitionSampling(PartitionSampling):
    pass
