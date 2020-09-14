from boomer.common._predictions cimport Prediction


cdef class PostProcessor:

    # Functions:

    cdef void post_process(self, Prediction* prediction)
