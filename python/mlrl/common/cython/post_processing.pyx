"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_shared


cdef class PostProcessor:
    """
    A wrapper for the pure virtual C++ class `IPostProcessor`.
    """
    pass


cdef class NoPostProcessor(PostProcessor):
    """
    A wrapper for the C++ class `NoPostProcessor`.
    """

    def __cinit__(self):
        self.post_processor_ptr = <shared_ptr[IPostProcessor]>make_shared[NoPostProcessorImpl]()
