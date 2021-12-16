"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport unique_ptr, make_unique


cdef class ConstantShrinkage(PostProcessor):
    """
    A wrapper for the C++ class `ConstantShrinkage`.
    """

    def __cinit__(self, float64 shrinkage):
        """
        :param shrinkage: The shrinkage parameter. Must be in (0, 1)
        """
        self.post_processor_ptr = <unique_ptr[IPostProcessor]>make_unique[ConstantShrinkageImpl](shrinkage)
