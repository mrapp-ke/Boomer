"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport shared_ptr, make_shared


cdef class ConstantShrinkage(PostProcessor):
    """
    A wrapper for the C++ class `ConstantShrinkage`.
    """

    def __cinit__(self, float64 shrinkage = 0.3):
        """
        :param shrinkage: The shrinkage parameter. Must be in (0, 1)
        """
        self.post_processor_ptr = <shared_ptr[IPostProcessor]>make_shared[ConstantShrinkageImpl](shrinkage)
