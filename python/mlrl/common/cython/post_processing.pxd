from libcpp.memory cimport unique_ptr


cdef extern from "common/post_processing/post_processor.hpp" nogil:

    cdef cppclass IPostProcessor:
        pass


cdef extern from "common/post_processing/post_processor_no.hpp" nogil:

    cdef cppclass NoPostProcessorImpl"NoPostProcessor"(IPostProcessor):
        pass


cdef class PostProcessor:

    # Attributes:

    cdef unique_ptr[IPostProcessor] post_processor_ptr


cdef class NoPostProcessor(PostProcessor):
    pass
