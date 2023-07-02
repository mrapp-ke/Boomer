from mlrl.common.cython._types cimport float32, uint32


cdef extern from "common/sampling/feature_sampling_without_replacement.hpp" nogil:

    cdef cppclass IFeatureSamplingWithoutReplacementConfig:

        # Functions:

        float32 getSampleSize() const

        IFeatureSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize) except +

        uint32 getNumRetained() const

        IFeatureSamplingWithoutReplacementConfig& setNumRetained(uint32 numRetained) except +


cdef class FeatureSamplingWithoutReplacementConfig:

    # Attributes:

    cdef IFeatureSamplingWithoutReplacementConfig* config_ptr
