from mlrl.common.cython._types cimport float64
from mlrl.common.cython.post_processing cimport PostProcessor, IPostProcessor


cdef extern from "boosting/post_processing/shrinkage_constant.hpp" namespace "boosting" nogil:

    cdef cppclass ConstantShrinkageImpl"boosting::ConstantShrinkage"(IPostProcessor):

        # Constructors:

        ConstantShrinkageImpl(float64 shrinkage) except +


cdef class ConstantShrinkage(PostProcessor):
    pass
