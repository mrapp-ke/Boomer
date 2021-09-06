from libcpp.memory cimport unique_ptr


cdef extern from "common/pruning/pruning.hpp" nogil:

    cdef cppclass IPruning:
        pass


cdef extern from "common/pruning/pruning_no.hpp" nogil:

    cdef cppclass NoPruningImpl"NoPruning"(IPruning):
        pass


cdef extern from "common/pruning/pruning_irep.hpp" nogil:

    cdef cppclass IREPImpl"IREP"(IPruning):
        pass


cdef class Pruning:

    # Attributes:

    cdef unique_ptr[IPruning] pruning_ptr


cdef class NoPruning(Pruning):
    pass


cdef class IREP(Pruning):
    pass
