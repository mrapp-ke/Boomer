"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_unique


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
        self.post_processor_ptr = <unique_ptr[IPostProcessor]>make_unique[NoPostProcessorImpl]()
