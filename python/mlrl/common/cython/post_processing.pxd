from libcpp.memory cimport shared_ptr


cdef extern from "common/post_processing/post_processor.hpp" nogil:

    cdef cppclass IPostProcessor:
        pass


cdef extern from "common/post_processing/post_processor_no.hpp" nogil:

    cdef cppclass NoPostProcessorImpl"NoPostProcessor"(IPostProcessor):
        pass


cdef class PostProcessor:

    # Attributes:

    cdef shared_ptr[IPostProcessor] post_processor_ptr


cdef class NoPostProcessor(PostProcessor):
    pass
