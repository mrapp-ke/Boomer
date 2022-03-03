from mlrl.common.cython._types cimport float64


cdef extern from "boosting/post_processing/shrinkage_constant.hpp" namespace "boosting" nogil:

    cdef cppclass IConstantShrinkageConfig:

        # Functions:

        float64 getShrinkage() const

        IConstantShrinkageConfig& setShrinkage(float64 shrinkage) except +


cdef class ConstantShrinkageConfig:

    # Attributes:

    cdef IConstantShrinkageConfig* config_ptr
