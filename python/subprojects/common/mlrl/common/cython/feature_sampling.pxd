from mlrl.common.cython._types cimport float32


cdef extern from "common/sampling/feature_sampling_without_replacement.hpp" nogil:

    cdef cppclass IFeatureSamplingWithoutReplacementConfig:

        # Functions:

        float32 getSampleSize() const

        IFeatureSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize) except +


cdef class FeatureSamplingWithoutReplacementConfig:

    # Attributes:

    cdef IFeatureSamplingWithoutReplacementConfig* config_ptr
