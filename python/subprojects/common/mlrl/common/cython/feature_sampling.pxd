from mlrl.common.cython._types cimport float32

from libcpp.memory cimport unique_ptr


cdef extern from "common/sampling/feature_sampling.hpp" nogil:

    cdef cppclass IFeatureSamplingFactory:
        pass


cdef extern from "common/sampling/feature_sampling_without_replacement.hpp" nogil:

    cdef cppclass FeatureSamplingWithoutReplacementFactoryImpl"FeatureSamplingWithoutReplacementFactory"(
            IFeatureSamplingFactory):

        # Constructors

        FeatureSamplingWithoutReplacementFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/feature_sampling_no.hpp" nogil:

    cdef cppclass NoFeatureSamplingFactoryImpl"NoFeatureSamplingFactory"(IFeatureSamplingFactory):
        pass


cdef class FeatureSamplingFactory:

    # Attributes:

    cdef unique_ptr[IFeatureSamplingFactory] feature_sampling_factory_ptr


cdef class FeatureSamplingWithoutReplacementFactory(FeatureSamplingFactory):
    pass


cdef class NoFeatureSamplingFactory(FeatureSamplingFactory):
    pass
