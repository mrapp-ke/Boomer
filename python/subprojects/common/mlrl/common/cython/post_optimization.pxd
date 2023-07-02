from libcpp cimport bool

from mlrl.common.cython._types cimport uint32


cdef extern from "common/post_optimization/post_optimization_sequential.hpp" nogil:

    cdef cppclass ISequentialPostOptimizationConfig:

        # Functions:

        uint32 getNumIterations() const

        ISequentialPostOptimizationConfig& setNumIterations(uint32 numIterations) except +

        bool areHeadsRefined() const

        ISequentialPostOptimizationConfig& setRefineHeads(bool refineHeads) except +

        bool areFeaturesResampled() const

        ISequentialPostOptimizationConfig& setResampleFeatures(bool resampleFeatures) except +


cdef class SequentialPostOptimizationConfig:

    # Attributes:

    cdef ISequentialPostOptimizationConfig* config_ptr
